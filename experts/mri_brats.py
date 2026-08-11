"""Brain-tumour MRI segmentation expert wrapping the MONAI Model Zoo `brats_mri_segmentation`
bundle (3D SegResNet, Apache-2.0) -- no training required.

Evaluated on the full 96-case MSD Task01_BrainTumour validation split before being wired
here: Dice TC=0.7931 / WT=0.8890 / ET=0.7506 / avg=0.8109 vs the bundle's own published
0.8559/0.9026/0.7905/0.8518 (see docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md). Contamination
status is UNVERIFIED -- the bundle's BraTS-2018 training split is not published in a form
that can be intersected with MSD Task01 case ids, so this Dice is evidence the pipeline is
wired correctly, not proof of held-out performance. That caveat is carried into every
`Finding` this expert produces (`extra["contamination_status"]`), not just documented here,
so it survives into the report/verifier trace rather than being forgotten downstream.

Unlike `TotalSegmentatorExpert` (one CT volume in, one label map out), this bundle needs
four co-registered MRI sequences -- T1c, T1, T2, FLAIR, in that exact channel order (its
own `configs/metadata.json` channel_def). A single `Scan.data` tensor cannot carry four
separate volumes, so `predict()` reads them from `scan.meta.extra["sequence_paths"]`, a
dict of on-disk paths keyed by sequence name (real spacing needed, same reasoning as
TotalSegmentator's `source_path`).

`api/persistence.py::run_analysis` now populates that key for study-level submissions
(`POST /v1/studies/{id}/analyses`) whose study has no single-file `storage_path` -- i.e. a
multi-series (currently: Orthanc-imported) study -- by grouping the study's `Series` rows
into the four canonical sequences via each series' own DICOM `SeriesDescription` tag (see
`api/persistence.py::_match_brats_sequences`). Two things worth knowing if you're
extending that path: (1) series-level submissions (`POST /v1/series/{id}/analyses`) still
only ever resolve one path, so they can never supply four sequences -- this expert is
unreachable from that endpoint by construction, not an oversight; (2) each path handed
into `sequence_paths` is a raw DICOM series directory, not a converted NIfTI file --
verified empirically that MONAI's `LoadImaged` reads a directory of same-series DICOM
instances directly (stacks correctly into a multi-channel volume), so no DICOM->NIfTI
conversion step was added. Because of that, spacing below is read from the *loaded*
image's own affine rather than via a second `nibabel.load()` of the reference path --
`nibabel` cannot open a DICOM directory at all, only a NIfTI file, so a spacing helper
that assumed NIfTI would silently break on this path.

TC/WT/ET are non-exclusive nested subregions (ET is inside TC is inside WT), not a single
argmax label map, so `Prediction.segmentation` holds three stacked boolean planes rather
than one integer label volume.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from reporting.findings import Finding

BRATS_BUNDLE_NAME = "brats_mri_segmentation"

# From the bundle's own configs/metadata.json channel_def: {0: T1c, 1: T1, 2: T2, 3: FLAIR}.
BUNDLE_MODALITY_ORDER: tuple[str, ...] = ("t1c", "t1", "t2", "flair")

BRATS_CHANNEL_NAMES: tuple[str, ...] = (
    "tumor core",
    "whole tumor",
    "enhancing tumor",
)
BRATS_CHANNEL_CODES: tuple[str, ...] = ("TC", "WT", "ET")

BRATS_REFERENCE_DICE = {"TC": 0.8559, "WT": 0.9026, "ET": 0.7905, "average": 0.8518}
OUR_VALIDATION_DICE = {"TC": 0.7931, "WT": 0.8890, "ET": 0.7506, "average": 0.8109}

_MODALITY_ALIASES = {
    "flair": "flair",
    "t2flair": "flair",
    "t1": "t1",
    "t1w": "t1",
    "t1n": "t1",
    "t1c": "t1c",
    "t1ce": "t1c",
    "t1gd": "t1c",
    "t2": "t2",
    "t2w": "t2",
    # Real clinical PACS `SeriesDescription` strings, verified with pydicom against
    # actual downloaded DICOM instances (not just index metadata) from UPENN-GBM-00020
    # (NCI Imaging Data Commons, collection `upenn_gbm`, CC BY 4.0) -- the DICOM-native
    # 4-sequence brain-tumour MRI case staged for this bundle's first real end-to-end
    # test; see docs/BRATS_TEST_CASE.md. As-shipped, `_MODALITY_ALIASES` matched none of
    # a real study's four series: no IDC brain-MRI collection surveyed (upenn_gbm,
    # icdc_glioma; tcga_gbm/tcga_lgg have no DICOM MR at all in IDC) uses the bare
    # canonical names above -- real scanners/post-processing pipelines stamp
    # protocol-specific strings instead. Kept as literal, collection-specific full
    # strings (including the "Processed_CaPTk" post-processing suffix) rather than
    # generalizing the parser (e.g. stripping known suffixes or fuzzy-matching
    # "axial"/"stealth" tokens): the same reasoning that keeps canonical_sequence_name
    # exact-match-only above applies here -- a fuzzy parser risks exactly the kind of
    # silent t1/t1gd-style misclassification this function was written to prevent, just
    # with a different pair of strings.
    "t1axial:processedcaptk": "t1",
    "t1axialstealthpost:processedcaptk": "t1c",
    "axialt2tse:processedcaptk": "t2",
    "t2flairaxial:processedcaptk": "flair",
}


def canonical_sequence_name(name: str) -> str:
    """Map a caller's MRI sequence label onto the bundle's vocabulary, or fail loudly.

    Exact lookup on a cleaned key, deliberately not prefix matching: 't1gd' starts with
    't1' but is the contrast-enhanced sequence -- confusing the two silently destroys the
    enhancing-tumour channel (the exact bug this project already hit once with MSD data).
    """
    key = re.sub(r"[\s_\-]", "", name.strip().lower())
    if key not in _MODALITY_ALIASES:
        raise ValueError(
            f"unrecognised MRI sequence name {name!r} (normalised {key!r}); refusing to "
            "guess which bundle input channel it belongs to"
        )
    return _MODALITY_ALIASES[key]


def resolve_sequence_paths(sequence_paths: Mapping[str, str]) -> list[str]:
    """Order caller-supplied {sequence_name: path} into the bundle's (t1c, t1, t2, flair).

    Raises if any of the four required sequences is missing, rather than silently
    proceeding with fewer channels -- a 3-channel input would run without error and
    produce a plausible-looking but meaningless prediction.
    """
    by_canonical: dict[str, str] = {}
    for name, path in sequence_paths.items():
        by_canonical[canonical_sequence_name(name)] = path
    missing = [m for m in BUNDLE_MODALITY_ORDER if m not in by_canonical]
    if missing:
        raise ValueError(
            f"missing MRI sequence(s) {missing}; brats_mri_segmentation needs all of "
            f"{list(BUNDLE_MODALITY_ORDER)}"
        )
    return [by_canonical[m] for m in BUNDLE_MODALITY_ORDER]


def findings_from_tc_wt_et_masks(
    masks: Sequence,
    spacing: tuple[float, ...] | None,
    *,
    min_volume_ml: float = 0.0,
    confidence: float | None = None,
    contamination_status: str = "UNVERIFIED",
) -> list[Finding]:
    """Build measured `Finding`s from three (D,H,W) boolean masks, in TC/WT/ET order.

    The pure, testable core -- takes plain arrays so a unit test can feed synthetic masks
    without touching MONAI/torch. Mirrors `ct_totalsegmentator.findings_from_label_counts`:
    volume in mL from voxel count x spacing, bounding-box extent in mm for `size_mm`.
    Without spacing, neither is reported (never fake physical units).
    """
    import numpy as np

    voxel_vol_ml = (math.prod(spacing) / 1000.0) if spacing else None
    findings: list[Finding] = []
    for code, label, mask in zip(BRATS_CHANNEL_CODES, BRATS_CHANNEL_NAMES, masks):
        arr = np.asarray(mask, dtype=bool)
        n = int(arr.sum())
        volume_ml = (n * voxel_vol_ml) if voxel_vol_ml is not None else None
        if volume_ml is not None and volume_ml < min_volume_ml:
            continue

        size_mm = None
        if n > 0 and spacing is not None and len(spacing) == arr.ndim:
            coords = np.argwhere(arr)
            extent = coords.max(0) - coords.min(0) + 1
            size_mm = float(max(e * s for e, s in zip(extent, spacing)))

        findings.append(
            Finding(
                label=label,
                probability=None,  # a thresholded segmentation, not a probability
                present=n > 0,
                confidence=confidence,
                volume_ml=volume_ml,
                size_mm=size_mm,
                source="mri-brats",
                canonical_label=code,
                extra={
                    "voxels": n,
                    "contamination_status": contamination_status,
                    "reference_dice": BRATS_REFERENCE_DICE,
                    "our_validation_dice": OUR_VALIDATION_DICE,
                },
            )
        )
    return findings


class BraTSExpert:
    """MONAI Model Zoo `brats_mri_segmentation` as a routable expert.

    Register one instance under (Modality.MRI, BodyPart.BRAIN). Needs a real bundle
    (downloaded once and cached under `bundle_root`) and four co-registered MRI sequences
    supplied via `scan.meta.extra["sequence_paths"]`.
    """

    def __init__(
        self,
        *,
        name: str = "mri_brats",
        bundle_root: str | Path,
        device: str | None = None,
        min_volume_ml: float = 0.1,
    ) -> None:
        self.name = name
        self.modality = Modality.MRI
        self.body_part = BodyPart.BRAIN
        self.bundle_root = Path(bundle_root)
        self.device_name = device
        self.min_volume_ml = float(min_volume_ml)
        self.version = f"{BRATS_BUNDLE_NAME}:monai-model-zoo"
        self.class_names = list(BRATS_CHANNEL_NAMES)
        self._network = None  # lazily built and cached on first predict()

    def predict(self, scan: Scan) -> Prediction:
        import torch
        from monai.inferers import sliding_window_inference
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
        )

        sequence_paths = (scan.meta.extra or {}).get("sequence_paths")
        if not sequence_paths:
            raise ValueError(
                "BraTSExpert needs scan.meta.extra['sequence_paths'] -- a dict of the four "
                "co-registered MRI sequences (t1c/t1/t2/flair) as on-disk NIfTI paths with "
                "real voxel spacing. Nothing was supplied."
            )
        ordered_paths = resolve_sequence_paths(sequence_paths)

        device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        network = self._load_network(device)

        transform = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        data = transform({"image": ordered_paths})
        loaded = data["image"]
        # Real spacing off the volume MONAI actually loaded, not a second read of
        # ordered_paths[0] -- that path may be a raw DICOM series directory (see the
        # module docstring), which `nibabel` cannot open at all.
        spacing = self._spacing_from_affine(getattr(loaded, "affine", None), loaded.ndim - 1)
        image = loaded.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=network,
                overlap=0.5,
            )
            masks = (torch.sigmoid(logits) > 0.5)[0].cpu().numpy().astype(bool)

        pred = Prediction(expert=self.name, meta=scan.meta)
        pred.segmentation = torch.as_tensor(masks)
        # The bundle exposes no single calibrated study-level confidence for a
        # segmentation; fabricating one would be indistinguishable from model evidence
        # downstream, so leave it explicitly unavailable (mirrors TotalSegmentatorExpert).
        pred.confidence = None
        pred.meta.extra = dict(pred.meta.extra or {})
        pred.meta.extra["seg_spacing"] = spacing
        return pred

    def findings_from_prediction(self, scan: Scan, pred: Prediction) -> list[Finding]:
        import numpy as np

        if pred.segmentation is None:
            return []
        seg = pred.segmentation
        arr = seg.detach().cpu().numpy() if hasattr(seg, "detach") else np.asarray(seg)
        spacing = pred.meta.extra.get("seg_spacing")
        return findings_from_tc_wt_et_masks(
            arr,
            spacing,
            min_volume_ml=self.min_volume_ml,
            confidence=pred.confidence,
        )

    # -- internals -----------------------------------------------------------
    def _load_network(self, device):
        """Instantiate and cache the bundle's own network, downloading it on first use.

        Cached on the instance (not per-call) since a fresh network build/checkpoint load
        per prediction would be wastefully slow for a repeatedly-registered live expert --
        the standalone script this was extracted from builds it once per process for the
        same reason.
        """
        if self._network is not None:
            return self._network

        import torch
        from monai.bundle import ConfigParser, download

        target = self.bundle_root / BRATS_BUNDLE_NAME
        if not (target / "configs" / "inference.json").is_file():
            self.bundle_root.mkdir(parents=True, exist_ok=True)
            download(name=BRATS_BUNDLE_NAME, bundle_dir=str(self.bundle_root))
        if not (target / "configs" / "inference.json").is_file():
            raise RuntimeError(
                f"bundle '{BRATS_BUNDLE_NAME}' downloaded but configs/inference.json is "
                f"missing at {target}"
            )

        parser = ConfigParser()
        parser.read_config(str(target / "configs" / "inference.json"))
        metadata_path = target / "configs" / "metadata.json"
        if metadata_path.is_file():
            parser.read_meta(str(metadata_path))

        network = None
        for key in ("network_def", "network"):
            try:
                network = parser.get_parsed_content(key, instantiate=True)
                break
            except Exception:  # noqa: BLE001 -- config key naming varies between bundles
                continue
        if network is None:
            raise RuntimeError(f"could not instantiate a network from {target}/configs")

        weights = target / "models" / "model.pt"
        if not weights.is_file():
            raise RuntimeError(f"bundle checkpoint missing: {weights}")
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if isinstance(state, dict) and not hasattr(state, "state_dict"):
            network.load_state_dict(state)
        else:  # a fully pickled module
            network = state
        self._network = network.to(device).eval()
        return self._network

    @staticmethod
    def _spacing_from_affine(affine, spatial_dims: int) -> tuple[float, ...] | None:
        """Physical voxel sizes from the norms of an affine's spatial basis vectors.

        Mirrors `ingest.loaders._spacing_from_affine` exactly (the diagonal shortcut is
        wrong for rotated volumes) -- duplicated rather than imported so this module
        doesn't need `ingest.loaders` for one three-line helper. Returns None only if
        the loader genuinely produced no affine (never fakes physical units).
        """
        import torch

        if affine is None:
            return None
        matrix = torch.as_tensor(affine, dtype=torch.float64)
        n_spatial = min(spatial_dims, 3, matrix.shape[1] - 1)
        return tuple(
            float(torch.linalg.vector_norm(matrix[:3, i])) for i in range(n_spatial)
        )
