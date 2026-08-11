"""Lung-nodule *detection* expert wrapping the MONAI Model Zoo `lung_nodule_ct_detection`
bundle (3D RetinaNet trained on LUNA16, Apache-2.0) -- no training required.

This is a purpose-built pathology detector, not an anatomy segmenter pressed into
service: `docs/DEVELOPMENT.md`'s CT-pathology section explains why TotalSegmentator's
`lung_nodules` task was replaced by this bundle (no published Dice/FROC for that task,
and it missed a clear 10mm expert-annotated nodule on this project's own LIDC test
case). The bundle's own validation-fold numbers -- mAP 0.852 / mAR 0.998 -- are carried
into every `Finding` this expert produces (`extra["reported_map"]`/`["reported_mar"]`),
not just documented here, mirroring `mri_brats.py`'s `reference_dice`/`our_validation_dice`
pattern: the caveat "this is the bundle's own number, not a re-derivation against this
project's data" must survive into the report/verifier trace, not get lost after this
module returns.

Boxes, not masks. RetinaNet is an object detector: each output is a 3D bounding box plus
one score, not a per-voxel label. Forcing that into `Prediction.segmentation` (a full
label volume) would either fabricate voxel-level boundaries the model never predicted, or
require rasterizing a box into a mask that looks more precise than it is. Instead
`predict()` leaves `segmentation=None` and stashes the raw detection list in
`pred.meta.extra["detections"]`; `findings_from_prediction()` is the only place that list
becomes `Finding`s, via the pure/testable `findings_from_detections()` helper (same split
as `ct_totalsegmentator.py`'s `findings_from_label_counts`).

Score threshold: the bundle's own `configs/inference.json` box selector is deliberately
loose (`score_thresh=0.02`, up to 300 candidates/image, kept for mAP curve computation
across many thresholds). Reporting all of those as findings would flood a study with
near-noise detections. `LUNG_NODULE_SCORE_THRESHOLD = 0.3` is not re-derived here -- it is
the exact operating point `scripts/run_monai_pathology_experts.py` used when matching
this same detector's boxes against LIDC consensus ground truth, so it is the threshold
this bundle has actually been evaluated at, not a guess.

DICOM -> NIfTI conversion: the bundle's `LoadImaged(reader="itkreader")` preprocessing
expects one volume file, and the live API hands every expert a `Scan` whose
`meta.source_path` is a bare DICOM series directory (`STORAGE_DIR/studies/{id}/series/{id}/`,
plain per-instance `.dcm` files -- confirmed by reading `api/dicom_ingest.py` and
`api/persistence.py::run_analysis`, which passes `series.storage_dir` straight into
`Pipeline.analyze`; `ingest/loaders.py::VolumeLoader` treats any directory as exactly
that). Unlike `TotalSegmentatorExpert`, which can hand that same directory straight to
`totalsegmentator()`, this bundle's `itkreader`-based transform chain is not a DICOM
series assembler, so `predict()` converts the directory to a scratch NIfTI file first,
via `SimpleITK.ImageSeriesReader` (LPS-correct), the same approach validated in
`scripts/run_monai_pathology_experts.py::_ct_dicom_to_nifti`. If `source_path` is already
a `.nii`/`.nii.gz` file, no conversion happens.

Nothing here is trained locally, and the bundle's mAP/mAR is a number from its own
LUNA16 validation fold, not evidence about performance on a target CT population -- that
requires geometry-preserving evaluation on held-out, contamination-checked data (see
`docs/AI_HANDOFF.md` section 6 and `docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md`). The heavy
deps (`torch`, `monai`, `SimpleITK`) are imported lazily so importing this module stays
cheap offline.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from reporting.findings import Finding

LUNG_NODULE_BUNDLE_NAME = "lung_nodule_ct_detection"

# The bundle's own reported validation-fold metrics (its model card / the run recorded
# in scripts/run_monai_pathology_experts.py) -- not re-derived here. Carried into every
# Finding's `extra` so this caveat survives downstream rather than being forgotten once
# this module returns.
LUNG_NODULE_REPORTED_MAP = 0.852
LUNG_NODULE_REPORTED_MAR = 0.998

# The operating point scripts/run_monai_pathology_experts.py used for hit/miss matching
# against LIDC consensus ground truth -- see module docstring. The detector's own box
# selector keeps candidates down to score_thresh=0.02; this is the higher bar findings
# are actually filtered at.
LUNG_NODULE_SCORE_THRESHOLD = 0.3


def findings_from_detections(
    detections: Sequence[Mapping],
    *,
    min_score: float = LUNG_NODULE_SCORE_THRESHOLD,
) -> list[Finding]:
    """Build `Finding`s from raw detector output (the pure, testable core).

    Each `detections` entry is `{"center_lps_mm": (x,y,z), "size_whd_mm": (w,h,d),
    "score": float}` -- exactly `_run_lung_nodule_detector`'s return shape in
    `scripts/run_monai_pathology_experts.py`. Detections below `min_score` are dropped
    (see module docstring for why the raw box-selector threshold is too loose to report
    directly). Only present findings are emitted -- there is no fixed vocabulary of
    "expected but absent" nodules the way BraTS has fixed TC/WT/ET channels, so an empty
    result means no candidate cleared `min_score`, not a proven-clear lung.
    """
    findings: list[Finding] = []
    for det in detections:
        score = float(det["score"])
        if score < min_score:
            continue

        w, h, d = (float(v) for v in det["size_whd_mm"])
        size_mm = max(w, h, d)
        # Ellipsoid volume, not bounding-box volume: the detector's box circumscribes a
        # roughly round/ellipsoid nodule, and a sphere inscribed in a cube fills only
        # pi/6 (~52%) of the cube's volume -- reporting box volume (w*h*d) would overstate
        # true nodule volume by a factor of ~1.91x. Treating the box's three edges as the
        # ellipsoid's three diameters is the more physically defensible estimate.
        volume_ml = (4.0 / 3.0) * math.pi * (w / 2.0) * (h / 2.0) * (d / 2.0) / 1000.0
        center_lps_mm = tuple(float(v) for v in det["center_lps_mm"])

        findings.append(
            Finding(
                label="pulmonary nodule",
                # The detector's per-box score is its own model output for "is this a
                # real nodule", directly analogous to a classification score (compare
                # findings_from_classification's probability=float(prob)) -- not a
                # separate, caller-fitted reliability estimate, so it belongs in
                # `probability`, not `confidence` (which mri_brats.py/ct_totalsegmentator.py
                # both leave None precisely because no calibrated score exists there).
                probability=score,
                present=True,
                confidence=None,
                size_mm=size_mm,
                volume_ml=volume_ml,
                source="ct-lung-nodule",
                canonical_label="nodule",
                extra={
                    "center_lps_mm": center_lps_mm,
                    "size_whd_mm": (w, h, d),
                    "reported_map": LUNG_NODULE_REPORTED_MAP,
                    "reported_mar": LUNG_NODULE_REPORTED_MAR,
                },
            )
        )
    findings.sort(key=lambda f: f.probability, reverse=True)
    return findings


class LungNoduleDetectorExpert:
    """MONAI Model Zoo `lung_nodule_ct_detection` (3D RetinaNet on LUNA16) as a routable
    expert. Register one instance under (Modality.CT, BodyPart.CHEST). Needs a real
    bundle (downloaded once and cached under `bundle_root`, same pattern as `BraTSExpert`).
    """

    def __init__(
        self,
        *,
        name: str = "ct_lung_nodule",
        bundle_root: str | Path,
        device: str | None = None,
        min_score: float = LUNG_NODULE_SCORE_THRESHOLD,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.CT
        self.body_part = BodyPart.CHEST
        self.bundle_root = Path(bundle_root)
        self.device_name = device
        self.min_score = float(min_score)
        # Scratch location for DICOM->NIfTI conversions (see module docstring). Not the
        # bundle cache -- this holds per-scan converted volumes, not model weights.
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / "doctor_assistant_ct_lung_nodule"
        )
        self.version = f"{LUNG_NODULE_BUNDLE_NAME}:monai-model-zoo"
        self.class_names = ["pulmonary nodule"]
        self._detector = None  # lazily built and cached on first predict()

    def predict(self, scan: Scan) -> Prediction:
        import torch

        source_path = scan.meta.source_path
        if not source_path:
            raise ValueError(
                "LungNoduleDetectorExpert needs scan.meta.source_path -- the on-disk CT "
                "(NIfTI file or DICOM series directory) with real voxel spacing."
            )

        device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        detector = self._load_detector(device)
        nifti_path = self._ensure_nifti(source_path)
        detections = self._run_detector(nifti_path, detector, device)

        pred = Prediction(expert=self.name, meta=scan.meta)
        # Boxes, not a mask -- see module docstring for why segmentation stays None.
        pred.segmentation = None
        # No single calibrated study-level confidence exists for a set of independent
        # per-box detections; fabricating one would be indistinguishable from model
        # evidence downstream (same reasoning as TotalSegmentatorExpert/BraTSExpert).
        pred.confidence = None
        pred.meta.extra = dict(pred.meta.extra or {})
        pred.meta.extra["detections"] = detections
        return pred

    def findings_from_prediction(self, scan: Scan, pred: Prediction) -> list[Finding]:
        detections = (pred.meta.extra or {}).get("detections") or []
        return findings_from_detections(detections, min_score=self.min_score)

    # -- internals -----------------------------------------------------------
    def _load_detector(self, device):
        """Instantiate and cache the RetinaNet detector, downloading the bundle on first
        use. Transcribed verbatim from
        `scripts/run_monai_pathology_experts.py::_load_lung_nodule_detector` -- hand-
        declared rather than run through MONAI's `ConfigParser` end-to-end because the
        bundle's config wires a full Ignite evaluator/handler pipeline through its own
        `scripts.*` package, unnecessary machinery for running the detector directly. If
        the bundle's own `configs/inference.json` ever changes these values, this needs
        to change with it. Cached on the instance for the same reason `BraTSExpert`
        caches its network: rebuilding per prediction would be wastefully slow for a
        repeatedly-registered live expert.
        """
        if self._detector is not None:
            return self._detector

        import torch
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
        from monai.apps.detection.networks.retinanet_network import (
            RetinaNet,
            resnet_fpn_feature_extractor,
        )
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.bundle import download
        from monai.networks.nets.resnet import resnet50

        bundle_dir = self.bundle_root / LUNG_NODULE_BUNDLE_NAME
        if not (bundle_dir / "models" / "model.pt").is_file():
            self.bundle_root.mkdir(parents=True, exist_ok=True)
            download(name=LUNG_NODULE_BUNDLE_NAME, bundle_dir=str(self.bundle_root))
        if not (bundle_dir / "models" / "model.pt").is_file():
            raise RuntimeError(
                f"bundle '{LUNG_NODULE_BUNDLE_NAME}' downloaded but models/model.pt is "
                f"missing at {bundle_dir}"
            )

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2, 4),
            base_anchor_shapes=((6, 8, 4), (8, 6, 5), (10, 10, 6)),
        )
        backbone = resnet50(
            spatial_dims=3, n_input_channels=1, conv1_t_stride=(2, 2, 1), conv1_t_size=(7, 7, 7)
        )
        feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
        network = RetinaNet(
            spatial_dims=3,
            num_classes=1,
            num_anchors=3,
            feature_extractor=feature_extractor,
            size_divisible=(16, 16, 8),
            use_list_output=False,
        ).to(device)

        checkpoint = torch.load(
            bundle_dir / "models" / "model.pt", map_location="cpu", weights_only=False
        )
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        network.load_state_dict(state)
        network.eval()

        detector = RetinaNetDetector(
            network=network,
            anchor_generator=anchor_generator,
            debug=False,
            spatial_dims=3,
            num_classes=1,
            size_divisible=(16, 16, 8),
        )
        detector.set_target_keys(box_key="box", label_key="label")
        detector.set_box_selector_parameters(
            score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.22, detections_per_img=300
        )
        detector.set_sliding_window_inferer(
            roi_size=(512, 512, 192), overlap=0.25, sw_batch_size=1, mode="constant", device="cpu"
        )
        # RetinaNetDetector is its own nn.Module with its own .training flag, separate
        # from the wrapped network's -- network.eval() alone does not touch it; without
        # this, inference-only calls raise "Please provide ground truth targets during
        # training." (the exact bug this was extracted from already hit once).
        detector.eval()
        self._detector = detector.to(device)
        return self._detector

    def _ensure_nifti(self, source_path: str) -> Path:
        """Return a NIfTI path for `source_path`, converting a DICOM series directory
        first if needed. See module docstring for why the live API hands this expert a
        DICOM directory rather than a NIfTI file.
        """
        src = Path(source_path)
        if src.is_file() and src.name.lower().endswith((".nii", ".nii.gz")):
            return src
        if src.is_dir():
            return self._dicom_dir_to_nifti(src)
        raise ValueError(
            f"LungNoduleDetectorExpert cannot read {source_path!r} -- expected a "
            "NIfTI file or a DICOM series directory."
        )

    def _dicom_dir_to_nifti(self, dicom_dir: Path) -> Path:
        """Assemble a DICOM series directory into one NIfTI volume via SimpleITK's
        `ImageSeriesReader` (LPS-correct), the same approach validated in
        `scripts/run_monai_pathology_experts.py::_ct_dicom_to_nifti`. Auto-discovers the
        series UID rather than requiring the caller to supply one, because a `Scan`'s
        `source_path` carries no series UID (see `ingest/loaders.py::VolumeLoader`) --
        `api/dicom_ingest.py` already guarantees one series' files per directory, so this
        fails loudly instead of silently picking one if that ever stops being true.
        """
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            raise RuntimeError(f"no DICOM series found under {dicom_dir}")
        if len(series_ids) > 1:
            raise RuntimeError(
                f"expected exactly one DICOM series under {dicom_dir}, found "
                f"{len(series_ids)}; refusing to guess which one to convert"
            )
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        if not files:
            raise RuntimeError(f"GDCM found no files for series {series_ids[0]} under {dicom_dir}")
        reader.SetFileNames(files)
        image = reader.Execute()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(dicom_dir).encode("utf-8")).hexdigest()[:16]
        out_path = self.cache_dir / f"{digest}.nii.gz"
        sitk.WriteImage(image, str(out_path))
        return out_path

    @staticmethod
    def _run_detector(nifti_path: Path, detector, device) -> list[dict]:
        """Preprocess exactly as the bundle's `inference.json` does, run the detector,
        and convert predicted boxes to world (LPS) millimetre coordinates. Transcribed
        verbatim from
        `scripts/run_monai_pathology_experts.py::_run_lung_nodule_detector` -- reuses
        MONAI's own transform classes for the LPS/RAS coordinate math rather than
        hand-rolling it, deliberately: getting that subtly wrong would silently
        misplace every predicted box with no local way to catch it.
        """
        import torch
        from monai.apps.detection.transforms.dictionary import (
            AffineBoxToWorldCoordinated,
            ClipBoxToImaged,
            ConvertBoxModed,
        )
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            LoadImaged,
            Orientationd,
            ScaleIntensityRanged,
            Spacingd,
        )

        preprocessing = Compose(
            [
                LoadImaged(keys="image", reader="itkreader", affine_lps_to_ras=True),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=(0.703125, 0.703125, 1.25)),
                ScaleIntensityRanged(
                    keys="image", a_min=-1024.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
                ),
                EnsureTyped(keys="image"),
            ]
        )
        data = preprocessing({"image": str(nifti_path)})
        image = data["image"].to(device)

        with torch.no_grad():
            outputs = detector(input_images=[image], use_inferer=True)
        prediction = outputs[0]

        postprocessing = Compose(
            [
                ClipBoxToImaged(
                    box_keys="box", label_keys="label", box_ref_image_keys="image", remove_empty=True
                ),
                AffineBoxToWorldCoordinated(
                    box_keys="box", box_ref_image_keys="image", affine_lps_to_ras=True
                ),
                ConvertBoxModed(box_keys="box", src_mode="xyzxyz", dst_mode="cccwhd"),
            ]
        )
        merged = {**prediction, "image": data["image"]}
        world = postprocessing(merged)

        boxes = world["box"].detach().cpu().numpy()
        scores = world["label_scores"].detach().cpu().numpy()
        detections = []
        for box, score in zip(boxes, scores):
            cx, cy, cz, w, h, d = (float(v) for v in box)
            detections.append(
                {
                    "center_lps_mm": (cx, cy, cz),
                    "size_whd_mm": (w, h, d),
                    "score": float(score),
                }
            )
        detections.sort(key=lambda det: -det["score"])
        return detections
