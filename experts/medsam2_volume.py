"""Box-prompted full-volume segmentation backed by MedSAM2.

MedSAM2 treats an ordered stack of medical-image slices like a video: a box on one
frame identifies the object, then the memory-enabled predictor propagates that object
forward and backward through the stack.  This adapter intentionally exposes a small,
repository-owned interface around the upstream implementation so the API and its tests
do not depend on MedSAM2's command-line demo or DeepLesion-specific CSV layout.

The upstream code/checkpoint are not vendored.  Install the official repository and
point ``MEDSAM2_CHECKPOINT_PATH`` at ``MedSAM2_latest.pt``; see
``docs/MEDSAM2_3D_OHIF.md``.  The inference sequence below follows the official
``medsam2_infer_3D_CT.py`` path: 512px RGB frames, ImageNet normalization, a box on the
seed frame, then propagation in both directions.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile

from .medsam_interactive import clip_box_to_image


def largest_connected_component(mask):
    """Keep the largest 3D component, matching MedSAM2's official CT demo cleanup."""
    import numpy as np
    from scipy import ndimage

    arr = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(arr)
    if count == 0:
        return arr
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _hiera_pos_embed_sidecar_path(model_ref: str | Path) -> Path:
    """Where a raw-position-embedding sidecar would live for a converted checkpoint.

    ``mlx-sam-convert`` bakes the Hiera trunk's windowed position embedding into a
    single ``pos_embed_full`` tensor sized for a 1024px image (a 256x256 patch grid),
    then the runtime slices its top-left corner to fit whatever ``image_size`` is
    actually configured. That slice is only correct when ``image_size`` is 1024; for
    MedSAM2's native 512px checkpoints it silently returns the position embedding for
    the wrong quadrant of the image. ``scripts/extract_hiera_pos_embed.py`` extracts the
    small raw tensors needed to recompute it correctly; this is where it looks for them.
    """
    path = Path(model_ref)
    suffix = "".join(path.suffixes) or ".safetensors"
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.pos_embed_raw.npz")


def _corrected_hiera_pos_embed_full(sidecar_path: Path, image_size: int, patch_stride: int = 4):
    """Recompute the Hiera trunk's ``pos_embed_full`` for the actual ``image_size``.

    Mirrors ``mlx_sam.convert.convert_state_dict``'s composition -- bicubic-upsample the
    small learned ``pos_embed`` to the patch grid, tile the learned ``pos_embed_window``
    across it, and add them -- but targets the patch grid MedSAM2 actually runs at
    (``image_size // patch_stride``) instead of the conversion script's hardcoded 256x256
    (which only matches a 1024px image).
    """
    import cv2
    import mlx.core as mx
    import numpy as np

    raw = np.load(sidecar_path)
    pos_embed = raw["pos_embed"][0]  # (C, 7, 7)
    pos_embed_window = raw["pos_embed_window"][0]  # (C, window, window)
    target = image_size // patch_stride
    channels = pos_embed.shape[0]
    upsampled = np.stack(
        [
            cv2.resize(pos_embed[c], (target, target), interpolation=cv2.INTER_CUBIC)
            for c in range(channels)
        ],
        axis=0,
    )
    window = pos_embed_window.shape[-1]
    if target % window != 0:
        raise ValueError(
            f"image_size {image_size} does not tile evenly with the {window}px "
            "windowed position embedding"
        )
    tiled = np.tile(pos_embed_window, (1, target // window, target // window))
    full = (upsampled + tiled).transpose(1, 2, 0)[None].astype(np.float32)
    return mx.array(full)


def _use_multimask_official(num_points: int) -> bool:
    """Whether MedSAM2's ``sam2.1_hiera_t512`` config would request multimask output.

    Mirrors ``SAM2Base._use_multimask`` in the official ``sam2`` package::

        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    The MedSAM2 CT checkpoint's config sets ``multimask_output_in_sam`` and
    ``multimask_output_for_tracking`` both to ``True`` and
    ``(multimask_min_pt_num, multimask_max_pt_num) = (0, 1)``, so with those two
    always-true terms the whole decision collapses to the point-count bound below,
    regardless of whether this is the initial (box) frame or a later propagation frame.
    """
    return 0 <= num_points <= 1


def _patch_mlx_predictor_multimask_selection(predictor) -> None:
    """Work around mlx-sam always using the wrong multimask branch of the mask decoder.

    ``mlx_sam.video_predictor.SAM2VideoPredictor`` hardcodes ``multimask_output=True``
    for the box-prompted seed frame (``_predict_initial``) and ``multimask_output=False``
    for every later propagation frame (``_predict_tracked``) -- the exact *opposite* of
    what the official ``sam2`` package's point-count-gated ``_use_multimask`` computes
    for a 2-point box prompt under MedSAM2's config (see ``_use_multimask_official``).
    Because the mask decoder's ``multimask_output`` flag also selects which output
    token is used for the object pointer (``mask_tokens_out[:, 0:1]`` for single-mask
    vs. an IoU-argmax pick among ``mask_tokens_out[:, 1:]`` for multimask), this
    silently swaps in the wrong SAM output token on every frame. Measured on
    LIDC-IDRI-0686 against the official Torch predictor on the same input: the
    predicted mask logits stay close (cosine ~0.999) because the hypernetwork/mask
    decode is fairly forgiving, but the projected object pointer -- which conditions
    memory attention on every later frame -- diverges sharply (cosine as low as -0.11,
    frame-to-frame). That is a plausible, evidenced mechanism for MLX losing the
    tracked object faster than Torch does as propagation moves away from the seed
    slice. This patches the two call sites, scoped to this predictor instance only, to
    use the official point-count rule instead of mlx-sam's hardcoded flags.
    """
    import types

    import mlx.core as mx
    import numpy as np

    def _predict_initial(self, encoded, points, labels):
        multimask_output = _use_multimask_official(int(labels.shape[0]))
        return self.model.predict_from_encoded(
            encoded,
            mx.array(points[None].astype(np.float32)),
            mx.array(labels[None].astype(np.int32)),
            multimask_output=multimask_output,
        )

    def _predict_tracked(
        self,
        encoded,
        obj_output_dict,
        frame_idx,
        point_inputs=None,
        prev_low=None,
        reverse: bool = False,
        memory_frame_idx=None,
    ):
        num_points = 0 if point_inputs is None else int(point_inputs["point_labels"].shape[0])
        multimask_output = _use_multimask_official(num_points)
        cond = list(obj_output_dict["cond_frame_outputs"].values())
        mem = list(obj_output_dict["non_cond_frame_outputs"].values())
        current_frame_idx = frame_idx if memory_frame_idx is None else memory_frame_idx
        conditioned = self.model.condition_with_memories(
            encoded,
            mem,
            cond_memories=cond,
            current_frame_idx=current_frame_idx,
            track_in_reverse=reverse,
        )
        conditioned_encoded = dict(encoded)
        conditioned_encoded["vision_features"] = conditioned
        if point_inputs is None:
            coords = labels = None
        else:
            coords = mx.array(point_inputs["point_coords"][None].astype(np.float32))
            labels = mx.array(point_inputs["point_labels"][None].astype(np.int32))
        return self.model.predict_from_encoded(
            conditioned_encoded,
            coords,
            labels,
            mask_input=prev_low if prev_low is not None else None,
            multimask_output=multimask_output,
            add_no_mem_embed=False,
        )

    predictor._predict_initial = types.MethodType(_predict_initial, predictor)
    predictor._predict_tracked = types.MethodType(_predict_tracked, predictor)


def prompt_connected_component(mask, seed_index: int, box_xyxy):
    """Keep the 3D component most strongly supported inside the seed-slice box.

    A video segmenter can hallucinate a large disconnected region far from the prompt.
    Selecting the globally largest component can then delete the actual prompt result,
    so interactive cleanup must remain anchored to the user's box.
    """
    import numpy as np
    from scipy import ndimage

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or not 0 <= seed_index < arr.shape[0]:
        raise ValueError("prompt component cleanup requires a valid 3D mask and seed index")
    labels, count = ndimage.label(arr, structure=ndimage.generate_binary_structure(3, 3))
    if count == 0:
        return arr
    x0, y0, x1, y1 = box_xyxy
    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(arr.shape[2], int(np.ceil(x1)))
    y1 = min(arr.shape[1], int(np.ceil(y1)))
    prompt_labels = labels[seed_index, y0:y1, x0:x1]
    candidates, votes = np.unique(prompt_labels[prompt_labels > 0], return_counts=True)
    if not candidates.size:
        raise ValueError("the predicted mask does not intersect the prompt box on the seed slice")
    selected = int(candidates[int(np.argmax(votes))])
    return labels == selected


class MedSAM2VolumeSegmenter:
    """Load one MedSAM2 predictor and answer 3D box-prompted requests.

    ``volume`` is ``(slices, rows, columns)`` uint8 data.  ``seed_index`` identifies
    the slice on which ``box_xyxy`` was drawn.  The returned boolean array has exactly
    the same shape and order, which lets the caller map every mask back to its source
    SOP Instance UID without relying on filenames or viewer stack direction.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        model_config: str = "configs/sam2.1_hiera_t512.yaml",
        device: str | None = None,
        image_size: int = 512,
        keep_largest_component: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found at {self.checkpoint_path}"
            )
        if image_size < 64:
            raise ValueError("MedSAM2 image_size must be at least 64")
        self.model_config = model_config
        self.device_name = device
        self.image_size = int(image_size)
        self.keep_largest_component = keep_largest_component
        self.version = f"medsam2:{self.checkpoint_path.name}"
        self._predictor = None

    def segment_volume(
        self,
        volume,
        seed_index: int,
        box_xyxy: tuple[float, float, float, float],
    ):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        arr = np.asarray(volume)
        if arr.ndim != 3:
            raise ValueError(f"volume must have shape (slices, rows, columns), got {arr.shape}")
        depth, height, width = arr.shape
        if not 0 <= seed_index < depth:
            raise ValueError(f"seed_index {seed_index} is outside a {depth}-slice volume")
        box = clip_box_to_image(box_xyxy, width, height)

        predictor = self._load_predictor()
        device = self._device(torch)

        frames = torch.as_tensor(arr, dtype=torch.float32, device=device)[:, None]
        frames = functional.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).repeat(1, 3, 1, 1)
        frames = frames / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
        frames = (frames - mean) / std

        masks = np.zeros((depth, height, width), dtype=bool)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        inference_state = None
        try:
            with torch.inference_mode(), autocast:
                inference_state = predictor.init_state(frames, height, width)
                self._add_box(predictor, inference_state, seed_index, box)
                self._collect_masks(
                    predictor.propagate_in_video(inference_state), masks
                )

                # The upstream CT path resets only the tracking state between passes,
                # then re-adds the same prompt before walking toward earlier slices.
                predictor.reset_state(inference_state)
                self._add_box(predictor, inference_state, seed_index, box)
                self._collect_masks(
                    predictor.propagate_in_video(inference_state, reverse=True), masks
                )
        finally:
            if inference_state is not None:
                predictor.reset_state(inference_state)

        if not masks.any():
            raise ValueError("MedSAM2 returned an empty mask for the supplied box")
        if self.keep_largest_component:
            masks = largest_connected_component(masks)
        return masks

    @staticmethod
    def _add_box(predictor, inference_state, seed_index: int, box) -> None:
        import numpy as np

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_index,
            obj_id=1,
            box=np.asarray(box, dtype=np.float32),
        )

    @staticmethod
    def _collect_masks(propagation, destination) -> None:
        for frame_index, _object_ids, mask_logits in propagation:
            index = int(frame_index)
            if 0 <= index < destination.shape[0]:
                destination[index] = mask_logits[0].detach().float().cpu().numpy()[0] > 0.0

    def _device(self, torch):
        if self.device_name:
            return torch.device(self.device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from sam2.build_sam import build_sam2_video_predictor_npz
        except ImportError as exc:
            raise RuntimeError(
                "MedSAM2 is not installed; clone the official bowang-lab/MedSAM2 "
                "repository and run `pip install -e .` inside its environment"
            ) from exc

        self._predictor = build_sam2_video_predictor_npz(
            self.model_config,
            str(self.checkpoint_path),
        )
        return self._predictor


class SAM2MLXVolumeSegmenter:
    """Apple-Silicon-native SAM2/converted-MedSAM2 volume propagation via MLX.

    ``mlx-sam`` accepts either one of its published SAM2.1 MLX model IDs or a local
    ``.safetensors`` checkpoint. A MedSAM2 Torch checkpoint can be converted with its
    ``mlx-sam-convert`` command; until that converted checkpoint passes the parity gate
    documented in ``docs/MEDSAM2_3D_OHIF.md``, a stock SAM2 MLX model must be described
    as SAM2, not as medically fine-tuned MedSAM2.
    """

    def __init__(
        self,
        *,
        model: str | Path = "avbiswas/sam2.1-hiera-small-mlx-16bit",
        image_size: int = 512,
        keep_prompt_component: bool = True,
        precompute_image_features: bool = False,
    ) -> None:
        import importlib.util

        if importlib.util.find_spec("mlx_sam") is None:
            raise RuntimeError(
                "mlx-sam is not installed; install it on Apple Silicon with "
                "`python -m pip install mlx-sam`"
            )
        self.model_ref = str(model)
        self.image_size = int(image_size)
        self.keep_prompt_component = keep_prompt_component
        self.precompute_image_features = precompute_image_features
        self.version = f"sam2-mlx:{Path(self.model_ref).name}"
        self._predictor = None

    def segment_volume(
        self,
        volume,
        seed_index: int,
        box_xyxy: tuple[float, float, float, float],
    ):
        import numpy as np
        from PIL import Image

        arr = np.asarray(volume)
        if arr.ndim != 3:
            raise ValueError(f"volume must have shape (slices, rows, columns), got {arr.shape}")
        depth, height, width = arr.shape
        if not 0 <= seed_index < depth:
            raise ValueError(f"seed_index {seed_index} is outside a {depth}-slice volume")
        box = clip_box_to_image(box_xyxy, width, height)
        masks = np.zeros((depth, height, width), dtype=bool)
        predictor = self._load_predictor()

        # mlx-sam's public video API currently accepts a video path or a directory of
        # JPEG/PNG frames. A request-scoped directory keeps us on that supported API;
        # zero-padded filenames preserve the DICOM order exactly.
        with tempfile.TemporaryDirectory(prefix="doctor-assistant-sam2-mlx-") as tmp:
            frame_dir = Path(tmp)
            for index, frame in enumerate(arr):
                Image.fromarray(frame.astype(np.uint8)).convert("RGB").save(
                    frame_dir / f"frame_{index:06d}.png"
                )
            state = predictor.init_state(
                frame_dir,
                precompute_image_features=self.precompute_image_features,
            )
            try:
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=seed_index,
                    obj_id=1,
                    box=np.asarray(box, dtype=np.float32),
                )
                self._collect_numpy_masks(
                    predictor.propagate_in_video(
                        state, start_frame_idx=seed_index, reverse=False
                    ),
                    masks,
                )
                # Match the official MedSAM2 CT inference sequence: forward tracking
                # mutates memory state, so reset and re-seed before reverse tracking.
                predictor.reset_state(state)
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=seed_index,
                    obj_id=1,
                    box=np.asarray(box, dtype=np.float32),
                )
                self._collect_numpy_masks(
                    predictor.propagate_in_video(
                        state, start_frame_idx=seed_index, reverse=True
                    ),
                    masks,
                )
            finally:
                predictor.reset_state(state)

        if not masks.any():
            raise ValueError("SAM2 MLX returned an empty mask for the supplied box")
        if self.keep_prompt_component:
            masks = prompt_connected_component(masks, seed_index, box)
        return masks

    @staticmethod
    def _collect_numpy_masks(propagation, destination) -> None:
        import numpy as np

        for frame_index, _object_ids, mask_logits in propagation:
            index = int(frame_index)
            if 0 <= index < destination.shape[0]:
                logits = np.asarray(mask_logits)
                destination[index] = logits[0, 0] > 0.0

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from mlx_sam import SAM2VideoPredictor

        predictor = SAM2VideoPredictor.from_pretrained(
            self.model_ref,
            image_size=self.image_size,
            memory_dtype="float16",
            memory_attention_dtype="float16",
        )
        self._patch_hiera_pos_embed(predictor)
        _patch_mlx_predictor_multimask_selection(predictor)
        self._predictor = predictor
        return self._predictor

    def _patch_hiera_pos_embed(self, predictor) -> None:
        """Work around mlx-sam baking the Hiera position embedding for a 1024px image.

        ``mlx-sam-convert`` precomputes the trunk's windowed position embedding as a
        256x256 patch grid (matching a 1024px input at stride 4) and the runtime just
        slices its top-left corner to whatever ``image_size`` is configured. That slice
        is the position embedding for the wrong quadrant of the image whenever
        ``image_size`` is not 1024 -- silently, since shapes still line up. MedSAM2's CT
        checkpoint runs at 512px, so every box-prompted volume segmentation was
        conditioning on corrupted positional information (measured: cosine similarity
        ~0.91, relative L2 ~0.42 on the raw image-encoder features vs. the official
        Torch checkpoint on the same input). If a sidecar with the small raw tensors is
        available, recompute the position embedding correctly for the real image_size.
        """
        target = self.image_size // 4
        current = predictor.model.trunk.pos_embed_full
        if current.shape[1] == target and current.shape[2] == target:
            return  # already the right grid size (e.g. the stock 1024px SAM2.1 models)
        sidecar = _hiera_pos_embed_sidecar_path(self.model_ref)
        if not sidecar.is_file():
            return  # nothing we can do without the raw tensors; leave prior behavior
        predictor.model.trunk.pos_embed_full = _corrected_hiera_pos_embed_full(
            sidecar, self.image_size
        )
