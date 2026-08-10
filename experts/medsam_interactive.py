"""Interactive, on-demand 2D segmentation backed by MedSAM (Ma et al., Apache-2.0,
https://github.com/bowang-lab/MedSAM) -- a SAM ViT-B fine-tuned on medical images.

This is deliberately NOT an `ExpertModel`: it has no `predict(scan)` and produces no
automatic per-scan result. It exists to answer one on-demand request -- "segment
whatever is inside this box, on this one 2D slice" -- triggered by a radiologist
interacting with the viewer, not run automatically by the pipeline/router. Forcing the
`ExpertModel` Protocol's one-prediction-per-scan shape onto it would misrepresent it to
the registry, which assumes exactly that shape.

PROMPT TYPE: box, not point. MedSAM's own inference reference implementation
(MedSAM_Inference.py) always calls `prompt_encoder(points=None, boxes=box_torch,
masks=None)` -- the released checkpoint was fine-tuned exclusively on bounding-box
prompts. The underlying SAM architecture *can* accept a point prompt through the same
prompt encoder, but doing so with this checkpoint would be untested, off-label use: the
project's own rule (see docs/CHEST_CLASSIFIER_RESET.md, docs/MONAI_PATHOLOGY_EXPERTS_
RESULTS.md) is to never claim evidence a model wasn't actually shown to support. So the
interaction this backs is "draw a small box," not "click one point" -- same amount of
radiologist effort, but faithful to what MedSAM was actually validated on.

License: bowang-lab/MedSAM's repository LICENSE is Apache-2.0 (verified 2026-08-06); no
separate restrictive license was found for the released checkpoints. This is the same
evidentiary bar this project already applied to TotalSegmentator/BraTS (both Apache-2.0
MONAI Model Zoo bundles) -- stricter than KAD-512, whose checkpoint licensing remains an
open, unresolved release blocker.
"""

from __future__ import annotations

from pathlib import Path


def encode_binary_mask_rle(mask) -> dict:
    """COCO-style RLE: {size:[h,w], counts:[run,run,...]}, first run always background.

    Column-major (Fortran) order, matching the pycocotools convention, so a frontend
    decoder needs no special-casing. Pure function -- no numpy/model dependency beyond
    the array itself, kept import-light so it's cheap to unit test.
    """
    import numpy as np

    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    flat = arr.flatten(order="F")

    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    # The loop above always starts from `current = False`, so if the mask itself starts
    # on foreground the very first emitted run is already a 0-length background run --
    # decoders never have to special-case "starts true" separately.
    return {"size": [int(h), int(w)], "counts": counts}


def decode_binary_mask_rle(rle: dict):
    """Inverse of `encode_binary_mask_rle`, for round-trip testing."""
    import numpy as np

    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for run in rle["counts"]:
        if value:
            flat[pos : pos + run] = True
        pos += run
        value = not value
    return flat.reshape((h, w), order="F")


def clip_box_to_image(box_xyxy: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a caller-supplied box into valid pixel bounds; raise if it's degenerate.

    A box entirely or partly outside the image (e.g. from a stale click after the
    viewport panned/zoomed) must fail loudly, not silently segment the wrong region.
    """
    x0, y0, x1, y1 = box_xyxy
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(int(round(x0)), width - 1))
    x1 = max(0, min(int(round(x1)), width))
    y0 = max(0, min(int(round(y0)), height - 1))
    y1 = max(0, min(int(round(y1)), height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(
            f"box {box_xyxy} does not overlap a usable region of a {width}x{height} image "
            f"(clipped to [{x0},{y0},{x1},{y1}])"
        )
    return x0, y0, x1, y1


class MedSAMBoxSegmenter:
    """Loads the MedSAM checkpoint once and answers box-prompted segmentation requests.

    Backed by HuggingFace `transformers`' native SAM support (`SamModel`/`SamProcessor`),
    not Meta's raw `segment_anything` package: the actual official Apache-2.0 checkpoint
    (`wanglab/medsam-vit-base` on the Hugging Face Hub -- the bowang-lab authors' own org,
    verified 2026-08-06) is packaged as a `transformers`-format `pytorch_model.bin` +
    `config.json`, not a raw `segment_anything`-loadable state dict. `transformers` is
    already a dependency here (MAIRA-2/KAD use it), and has first-class, well-documented
    SAM support, so this is the correct loader for this checkpoint, not a workaround.

    Long-lived: construct once at API startup (mirrors `api/registry.py`'s experts,
    which are also built once and reused), not per-request -- the ViT-B image encoder is
    the expensive part and there is no reason to pay for it twice.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        model_id: str = "wanglab/medsam-vit-base",
        device: str | None = None,
    ) -> None:
        # `checkpoint_path` (a local directory containing config.json/pytorch_model.bin,
        # e.g. from `huggingface-cli download wanglab/medsam-vit-base --local-dir ...`)
        # takes precedence when given, so a fully offline deployment never needs network
        # access at request time; `model_id` falls back to the Hub, downloading/caching
        # via `transformers`' own mechanism on first use.
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
            if not self.checkpoint_path.is_dir():
                raise FileNotFoundError(f"MedSAM checkpoint directory not found at {self.checkpoint_path}")
            self._pretrained_ref = str(self.checkpoint_path)
        else:
            self.checkpoint_path = None
            self._pretrained_ref = model_id
        self.device_name = device
        self.version = f"medsam:{self._pretrained_ref}"
        self._model = None  # lazily built and cached on first segment_box()
        self._processor = None

    def segment_box(self, image, box_xyxy: tuple[float, float, float, float]):
        """Segment the region inside `box_xyxy` on a single 2D grayscale/RGB slice.

        `image`: a 2D (H,W) or (H,W,3) array, already windowed to displayable intensity
        (this mirrors what the viewer is already showing the radiologist -- MedSAM's own
        preprocessing expects a roughly 0-255-range image, not raw HU/signal values).
        Returns a boolean (H,W) mask. Raises if the box doesn't overlap the image.
        """
        import numpy as np
        import torch
        from PIL import Image

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        height, width = arr.shape[:2]
        box = clip_box_to_image(box_xyxy, width, height)

        model = self._load_model()
        processor = self._load_processor()
        device = next(model.parameters()).device

        pil_image = Image.fromarray(arr.astype(np.uint8))
        inputs = processor(
            pil_image,
            input_boxes=[[list(box)]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # post_process_masks undoes the processor's own resize/pad back to the
        # original (height, width) -- the inverse of whatever `processor(...)` did, so
        # this never has to duplicate that resizing logic by hand.
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # (batch=1, num_boxes=1, num_preds=1, H, W) -> (H, W)
        mask = masks[0][0, 0].numpy().astype(bool)
        return mask

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch
        from transformers import SamModel

        device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = SamModel.from_pretrained(self._pretrained_ref)
        self._model = model.to(device).eval()
        return self._model

    def _load_processor(self):
        if self._processor is not None:
            return self._processor
        from transformers import SamProcessor

        self._processor = SamProcessor.from_pretrained(self._pretrained_ref)
        return self._processor
