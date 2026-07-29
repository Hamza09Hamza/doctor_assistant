"""KAD-512 zero-shot chest-X-ray classifier.

This module is a small, dependency-clean implementation of the inference path from
KAD (Knowledge-enhanced Visual-Language Pre-training on Chest Radiology Images).  It
intentionally does not import the original repository: that code performs global
object-storage setup at import time and depends on several training-only packages.

The three modules below retain the *official checkpoint key layout*:

* ``KADResNet512Encoder`` matches ``ModelRes512`` (``resnet``, ``res_features``,
  ``res_l1`` and ``res_l2``).
* ``KADTextEncoder`` matches ``CLP_clinical`` (``bert_model`` and ``mlp_embed``).
* ``KADQueryDecoder`` matches ``TQN_Model`` (the four custom decoder layers and
  two-class ``mlp_head``).

``KAD512Expert`` presents those components through this project's normal
``predict(Scan) -> Prediction`` interface.  Transformers and the tokenizer are loaded
only when a full KAD checkpoint is used.  For normal inference, export a query pack
once: it caches fixed prompt embeddings and omits BERT entirely.

KAD is a research model.  Its softmax values are ranking scores, not clinically
validated probabilities; operating thresholds still need to be selected on a
representative validation cohort.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan

KAD512_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

# These are the exact ChestX-ray14 queries used by the official KAD evaluator.
KAD512_PROMPTS: tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "pleural effusion",
    "infiltration",
    "lung mass",
    "lung nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural thicken",
    "hernia",
)

KAD512_IMAGE_SIZE = 512
KAD512_EMBED_DIM = 768
KAD512_BERT_MODEL_ID = "xmcmic/Med-KEBERT"
KAD512_BERT_REVISION = "3e877eb9119224b81eaffe13839cd93f073cfc37"
KAD512_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
KAD512_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_QUERY_PACK_FORMAT = "doctor_assistant.kad512.query_pack"
_QUERY_PACK_VERSION = 2


class KADCheckpointError(RuntimeError):
    """A KAD checkpoint or query pack is missing or incompatible."""


class KADResNet512Encoder(nn.Module):
    """ResNet-50 patch encoder with official KAD-512 state-dict names.

    ``imagenet_pretrained`` is off by default because a KAD checkpoint contains the
    complete ResNet.  Enabling it is useful only when intentionally initializing a
    new training run and may trigger a torchvision download.
    """

    def __init__(self, *, imagenet_pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        self.resnet = resnet50(weights=weights)
        num_features = int(self.resnet.fc.in_features)  # 2048 for ResNet-50

        # The original KAD module registers the same ResNet children under both
        # `resnet.*` and `res_features.*`.  Keeping that alias is deliberate: official
        # checkpoints contain both sets of keys.
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.res_l1 = nn.Linear(num_features, num_features)
        self.res_l2 = nn.Linear(num_features, KAD512_EMBED_DIM)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.res_features(image)
        patches = feature_map.flatten(2).transpose(1, 2)
        patches = self.res_l2(F.relu(self.res_l1(patches)))
        return patches, patches.mean(dim=1)


class KADTextEncoder(nn.Module):
    """Med-KEBERT wrapper with official KAD text-encoder state-dict names.

    The Hugging Face model is injected, which keeps ``transformers`` optional at
    module import time and lets compatibility tests use a tiny synthetic BERT.
    """

    def __init__(self, bert_model: nn.Module, *, embed_dim: int = KAD512_EMBED_DIM) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = int(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(()) * torch.log(torch.tensor(1 / 0.07)))
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.constant_(self.logit_scale, float(torch.log(torch.tensor(1 / 0.07))))
        for layer in self.mlp_embed:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=self.embed_dim**-0.5)

    def encode_text(self, tokens: Mapping[str, Tensor]) -> Tensor:
        output = self.bert_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        pooled = getattr(output, "pooler_output", None)
        if pooled is None and isinstance(output, (tuple, list)) and len(output) > 1:
            pooled = output[1]
        if pooled is None:
            raise RuntimeError(
                "KAD requires a BERT-style pooler output, but the configured text "
                "encoder returned none. Use the official xmcmic/Med-KEBERT architecture."
            )
        return self.mlp_embed(pooled)

    def forward(
        self,
        text1: Mapping[str, Tensor],
        text2: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        feature1 = F.normalize(self.encode_text(text1), dim=-1)
        feature2 = F.normalize(self.encode_text(text2), dim=-1)
        return feature1, feature2, self.logit_scale.exp()


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise ValueError(f"Unsupported KAD decoder activation: {name!r}")


class KADTransformerDecoderLayer(nn.Module):
    """The pre-norm decoder layer used by the released KAD checkpoint."""

    def __init__(
        self,
        d_model: int = KAD512_EMBED_DIM,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)
        self.normalize_before = bool(normalize_before)

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def _forward_pre(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        target_mask: Tensor | None,
        memory_mask: Tensor | None,
        target_key_padding_mask: Tensor | None,
        memory_key_padding_mask: Tensor | None,
        pos: Tensor | None,
        query_pos: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        target2 = self.norm1(target)
        query = key = self._with_pos(target2, query_pos)
        target2, _ = self.self_attn(
            query,
            key,
            value=target2,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        target = target + self.dropout1(target2)

        target2 = self.norm2(target)
        target2, attention = self.multihead_attn(
            query=self._with_pos(target2, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )
        target = target + self.dropout2(target2)

        target2 = self.norm3(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout3(target2)
        return target, attention

    def _forward_post(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        target_mask: Tensor | None,
        memory_mask: Tensor | None,
        target_key_padding_mask: Tensor | None,
        memory_key_padding_mask: Tensor | None,
        pos: Tensor | None,
        query_pos: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # Kept for state/behavior compatibility even though released KAD uses pre-norm.
        query = key = self._with_pos(target, query_pos)
        target2, _ = self.self_attn(
            query,
            key,
            value=target,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        target = self.norm1(target + self.dropout1(target2))
        target2, attention = self.multihead_attn(
            query=self._with_pos(target, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )
        target = self.norm2(target + self.dropout2(target2))
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout3(target2)), attention

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        method = self._forward_pre if self.normalize_before else self._forward_post
        return method(
            target,
            memory,
            target_mask=target_mask,
            memory_mask=memory_mask,
            target_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
        )


class KADTransformerDecoder(nn.Module):
    """Four cloned KAD decoder layers with the official ``decoder.*`` layout."""

    def __init__(
        self,
        decoder_layer: KADTransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = int(num_layers)
        self.norm = norm

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = target
        attention: Tensor | None = None
        for layer in self.layers:
            output, attention = layer(
                output,
                memory,
                target_mask,
                memory_mask,
                target_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        if attention is None:  # construction guarantees at least one layer
            raise RuntimeError("KAD query decoder has no layers.")
        return output, attention


class KADQueryDecoder(nn.Module):
    """Text-query decoder matching the released ``TQN_Model`` checkpoint."""

    def __init__(self, *, embed_dim: int = KAD512_EMBED_DIM, class_num: int = 2) -> None:
        super().__init__()
        self.d_model = int(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones(()) * torch.log(torch.tensor(1 / 0.07)))
        layer = KADTransformerDecoderLayer(
            self.d_model,
            4,
            1024,
            0.1,
            "relu",
            normalize_before=True,
        )
        self.decoder_norm = nn.LayerNorm(self.d_model)
        # The shared norm alias intentionally produces both `decoder_norm.*` and
        # `decoder.norm.*` state keys, exactly like the official model.
        self.decoder = KADTransformerDecoder(layer, 4, self.decoder_norm)
        self.dropout_feas = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if image_features.ndim != 3:
            raise ValueError(
                "image_features must have shape (batch, patches, 768), got "
                f"{tuple(image_features.shape)}"
            )
        if text_features.ndim != 2:
            raise ValueError(
                "text_features must have shape (queries, 768), got "
                f"{tuple(text_features.shape)}"
            )
        if image_features.shape[-1] != self.d_model or text_features.shape[-1] != self.d_model:
            raise ValueError(
                f"KAD feature width must be {self.d_model}; got image "
                f"{image_features.shape[-1]} and text {text_features.shape[-1]}."
            )

        batch_size = image_features.shape[0]
        memory = self.decoder_norm(image_features.transpose(0, 1))
        target = text_features.unsqueeze(1).expand(-1, batch_size, -1)
        target = self.decoder_norm(target)
        features, attention = self.decoder(target, memory)
        logits = self.mlp_head(self.dropout_feas(features).transpose(0, 1))
        return (logits, attention) if return_attention else logits


def preprocess_kad512(data: Any) -> Tensor:
    """Apply the released KAD NIH PIL-BICUBIC/ToTensor input contract.

    Canonical file inference passes a PIL image directly, matching the official
    ``Image.open(...).convert("RGB") -> Resize(BICUBIC) -> ToTensor`` path.
    Tensor inputs remain supported for the normal expert interface; they are
    converted to an 8-bit PIL image before the same canonical transform.
    """

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("KAD-512 preprocessing requires Pillow and NumPy") from exc

    value = data.as_tensor() if hasattr(data, "as_tensor") else data
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        tensor = torch.as_tensor(value).detach()
        original_dtype = tensor.dtype
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                "KAD-512 expects an unbatched (C,H,W) or (H,W) image, got "
                f"shape {tuple(tensor.shape)}."
            )
        if tensor.shape[0] not in (1, 3):
            raise ValueError(
                f"KAD-512 expects 1 or 3 channels, got {tensor.shape[0]}."
            )
        tensor = tensor.float()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("KAD-512 input contains NaN or infinite pixel values.")
        if original_dtype == torch.uint8:
            scaled = tensor
        else:
            lo, hi = float(tensor.min()), float(tensor.max())
            if lo >= 0.0 and hi <= 1.0:
                scaled = tensor * 255.0
            elif lo >= 0.0 and hi <= 255.0:
                scaled = tensor
            elif hi > lo:
                scaled = (tensor - lo) * (255.0 / (hi - lo))
            else:
                scaled = torch.zeros_like(tensor)
        byte_tensor = scaled.clamp(0.0, 255.0).round().to(torch.uint8)
        if byte_tensor.shape[0] == 1:
            array = byte_tensor[0].cpu().numpy()
            image = Image.fromarray(array, mode="L").convert("RGB")
        else:
            array = byte_tensor.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(array, mode="RGB")

    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(
        (KAD512_IMAGE_SIZE, KAD512_IMAGE_SIZE),
        resample=resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (KAD512_IMAGE_SIZE, KAD512_IMAGE_SIZE, 3):
        raise ValueError(f"KAD-512 PIL conversion returned invalid shape {array.shape}.")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1) / 255.0
    mean = tensor.new_tensor(KAD512_IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(KAD512_IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _normalized_state_dict(section: Any, name: str) -> dict[str, Tensor]:
    if not isinstance(section, Mapping):
        raise KADCheckpointError(
            f"KAD checkpoint section {name!r} must be a state-dict mapping, "
            f"got {type(section).__name__}."
        )
    state = dict(section)
    if not state:
        raise KADCheckpointError(f"KAD checkpoint section {name!r} is empty.")
    if not all(isinstance(key, str) for key in state):
        raise KADCheckpointError(f"KAD checkpoint section {name!r} has non-string keys.")

    # Accept common wrappers while retaining official names internally.
    for prefix in ("module.", "_orig_mod."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
    non_tensors = [key for key, value in state.items() if not isinstance(value, Tensor)]
    if non_tensors:
        shown = ", ".join(non_tensors[:3])
        raise KADCheckpointError(
            f"KAD checkpoint section {name!r} contains non-tensor entries: {shown}."
        )
    return state


def _require_shapes(
    section: str,
    state: Mapping[str, Tensor],
    required: Mapping[str, tuple[int | None, ...]],
) -> None:
    problems: list[str] = []
    for key, expected in required.items():
        value = state.get(key)
        if value is None:
            problems.append(f"missing {key}")
            continue
        actual = tuple(value.shape)
        if len(actual) != len(expected) or any(
            want is not None and got != want for got, want in zip(actual, expected)
        ):
            problems.append(f"{key} has shape {actual}, expected {expected}")
    if problems:
        detail = "; ".join(problems)
        if section == "image_encoder":
            res_l1 = state.get("res_l1.weight")
            if isinstance(res_l1, Tensor) and tuple(res_l1.shape) == (1024, 1024):
                detail += (
                    "; this looks like the lower-resolution KAD ResNet encoder, "
                    "not KAD-512 (which uses 2048-wide layer-4 features)"
                )
        raise KADCheckpointError(f"Incompatible KAD {section} state: {detail}.")


def preflight_kad512_checkpoint(checkpoint: Any) -> dict[str, dict[str, Tensor]]:
    """Validate and normalize the three sections in an official KAD-512 checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise KADCheckpointError(
            "KAD checkpoint must be a mapping with 'image_encoder', 'text_encoder', "
            f"and 'model' sections; got {type(checkpoint).__name__}."
        )
    required_sections = ("image_encoder", "text_encoder", "model")
    missing = [name for name in required_sections if name not in checkpoint]
    if missing:
        raise KADCheckpointError(
            "Not an official KAD inference checkpoint: missing top-level section(s) "
            + ", ".join(repr(name) for name in missing)
            + "."
        )

    sections = {
        name: _normalized_state_dict(checkpoint[name], name) for name in required_sections
    }
    _require_shapes(
        "image_encoder",
        sections["image_encoder"],
        {
            "resnet.conv1.weight": (64, 3, 7, 7),
            "res_features.7.2.conv3.weight": (2048, 512, 1, 1),
            "res_l1.weight": (2048, 2048),
            "res_l1.bias": (2048,),
            "res_l2.weight": (KAD512_EMBED_DIM, 2048),
            "res_l2.bias": (KAD512_EMBED_DIM,),
        },
    )
    _require_shapes(
        "text_encoder",
        sections["text_encoder"],
        {
            "bert_model.embeddings.word_embeddings.weight": (None, KAD512_EMBED_DIM),
            "mlp_embed.0.weight": (KAD512_EMBED_DIM, KAD512_EMBED_DIM),
            "mlp_embed.2.weight": (KAD512_EMBED_DIM, KAD512_EMBED_DIM),
            "logit_scale": (),
        },
    )
    _require_shapes(
        "model",
        sections["model"],
        {
            "decoder_norm.weight": (KAD512_EMBED_DIM,),
            "decoder.norm.weight": (KAD512_EMBED_DIM,),
            "decoder.layers.0.self_attn.in_proj_weight": (
                3 * KAD512_EMBED_DIM,
                KAD512_EMBED_DIM,
            ),
            "decoder.layers.3.linear2.weight": (KAD512_EMBED_DIM, 1024),
            "mlp_head.0.weight": (2, KAD512_EMBED_DIM),
            "mlp_head.0.bias": (2,),
            "logit_scale": (),
        },
    )
    return sections


def _validate_module_state(
    section: str,
    module: nn.Module,
    state: Mapping[str, Tensor],
) -> None:
    expected = module.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state)

    # Old transformers checkpoints sometimes persisted these deterministic buffers;
    # current versions regenerate them and intentionally omit them from state_dict().
    benign_text_buffers = {
        "bert_model.embeddings.position_ids",
        "bert_model.embeddings.token_type_ids",
    }
    allowed_extra = benign_text_buffers if section == "text_encoder" else set()
    allowed_missing = benign_text_buffers if section == "text_encoder" else set()
    missing = sorted((expected_keys - actual_keys) - allowed_missing)
    unexpected = sorted((actual_keys - expected_keys) - allowed_extra)
    mismatched = [
        f"{key}: checkpoint {tuple(state[key].shape)} != model {tuple(expected[key].shape)}"
        for key in sorted(expected_keys & actual_keys)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or mismatched:
        parts: list[str] = []
        if missing:
            parts.append("missing keys " + ", ".join(missing[:5]))
        if unexpected:
            parts.append("unexpected keys " + ", ".join(unexpected[:5]))
        if mismatched:
            parts.append("shape mismatches " + "; ".join(mismatched[:5]))
        raise KADCheckpointError(
            f"KAD {section} does not match the installed architecture: " + " | ".join(parts)
        )


def load_kad512_checkpoint(
    checkpoint: Any,
    *,
    image_encoder: KADResNet512Encoder,
    text_encoder: KADTextEncoder,
    query_decoder: KADQueryDecoder,
) -> None:
    """Preflight all components, then load atomically enough to avoid partial surprises."""

    sections = preflight_kad512_checkpoint(checkpoint)
    components: tuple[tuple[str, nn.Module], ...] = (
        ("image_encoder", image_encoder),
        ("text_encoder", text_encoder),
        ("model", query_decoder),
    )
    for section, module in components:
        _validate_module_state(section, module, sections[section])
    for section, module in components:
        # Known non-persistent BERT buffers are the only allowed incompatibility, and
        # were checked above; strict=False lets old/new transformers bridge that gap.
        module.load_state_dict(sections[section], strict=(section != "text_encoder"))


def _load_torch_mapping(
    path: str | Path,
    *,
    description: str,
    allow_unsafe_pickle: bool = False,
) -> Mapping[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise KADCheckpointError(f"{description} file does not exist: {resolved}")
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=not allow_unsafe_pickle,
        )
    except Exception as exc:
        hint = ""
        if not allow_unsafe_pickle:
            hint = (
                " If this is the trusted official release and it contains legacy "
                "pickled metadata, retry with allow_unsafe_pickle=True."
            )
        raise KADCheckpointError(f"Could not load {description} {resolved}: {exc}.{hint}") from exc
    if not isinstance(loaded, Mapping):
        raise KADCheckpointError(
            f"{description} {resolved} contained {type(loaded).__name__}, expected a mapping."
        )
    return loaded


def _query_pack_preprocess_metadata() -> dict[str, Any]:
    return {
        "image_size": KAD512_IMAGE_SIZE,
        "channels": 3,
        "color_mode": "RGB",
        "resize": "PIL.Image.resize square BICUBIC",
        "to_tensor": "uint8 RGB to float32 CHW divided by 255",
        "mean": list(KAD512_IMAGENET_MEAN),
        "std": list(KAD512_IMAGENET_STD),
    }


def build_kad512_query_pack(
    checkpoint: Any,
    *,
    text_features: Tensor,
    labels: Sequence[str],
    prompts: Sequence[str],
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a BERT-free inference artifact from a validated full checkpoint.

    ``text_features`` must have been produced by the loaded checkpoint's
    ``KADTextEncoder`` for ``prompts``.  The returned mapping contains only plain
    metadata and tensors, so it can be loaded with ``torch.load(weights_only=True)``.
    """

    sections = preflight_kad512_checkpoint(checkpoint)
    labels = tuple(str(value) for value in labels)
    prompts = tuple(str(value) for value in prompts)
    features = torch.as_tensor(text_features).detach().float().cpu()
    if not labels or len(labels) != len(prompts):
        raise KADCheckpointError("KAD query-pack labels and prompts must be non-empty and aligned.")
    if features.shape != (len(prompts), KAD512_EMBED_DIM):
        raise KADCheckpointError(
            "KAD query features must have shape "
            f"({len(prompts)}, {KAD512_EMBED_DIM}), got {tuple(features.shape)}."
        )
    if not bool(torch.isfinite(features).all()):
        raise KADCheckpointError("KAD query features contain NaN or infinite values.")

    return {
        "format": _QUERY_PACK_FORMAT,
        "format_version": _QUERY_PACK_VERSION,
        "image_encoder": sections["image_encoder"],
        "model": sections["model"],
        "text_features": features,
        "labels": list(labels),
        "prompts": list(prompts),
        "preprocess": _query_pack_preprocess_metadata(),
        "source": dict(source_metadata or {}),
    }


def preflight_kad512_query_pack(pack: Any) -> dict[str, Any]:
    """Validate a lean query pack without importing transformers."""

    if not isinstance(pack, Mapping):
        raise KADCheckpointError(
            f"KAD query pack must be a mapping, got {type(pack).__name__}."
        )
    if pack.get("format") != _QUERY_PACK_FORMAT:
        raise KADCheckpointError(
            f"Not a KAD-512 query pack: expected format {_QUERY_PACK_FORMAT!r}."
        )
    if pack.get("format_version") != _QUERY_PACK_VERSION:
        raise KADCheckpointError(
            "Unsupported KAD query-pack version "
            f"{pack.get('format_version')!r}; expected {_QUERY_PACK_VERSION}."
        )
    missing = [
        key
        for key in ("image_encoder", "model", "text_features", "labels", "prompts", "preprocess")
        if key not in pack
    ]
    if missing:
        raise KADCheckpointError("KAD query pack is missing: " + ", ".join(missing) + ".")

    image_state = _normalized_state_dict(pack["image_encoder"], "image_encoder")
    model_state = _normalized_state_dict(pack["model"], "model")
    _require_shapes(
        "image_encoder",
        image_state,
        {
            "resnet.conv1.weight": (64, 3, 7, 7),
            "res_features.7.2.conv3.weight": (2048, 512, 1, 1),
            "res_l1.weight": (2048, 2048),
            "res_l2.weight": (KAD512_EMBED_DIM, 2048),
        },
    )
    _require_shapes(
        "model",
        model_state,
        {
            "decoder.layers.0.self_attn.in_proj_weight": (
                3 * KAD512_EMBED_DIM,
                KAD512_EMBED_DIM,
            ),
            "decoder.layers.3.linear2.weight": (KAD512_EMBED_DIM, 1024),
            "mlp_head.0.weight": (2, KAD512_EMBED_DIM),
        },
    )

    labels = pack["labels"]
    prompts = pack["prompts"]
    if (
        not isinstance(labels, (list, tuple))
        or not isinstance(prompts, (list, tuple))
        or not labels
        or len(labels) != len(prompts)
        or not all(isinstance(value, str) and value for value in (*labels, *prompts))
    ):
        raise KADCheckpointError("KAD query-pack labels/prompts are invalid or misaligned.")
    features = pack["text_features"]
    if not isinstance(features, Tensor) or tuple(features.shape) != (
        len(prompts),
        KAD512_EMBED_DIM,
    ):
        shape = tuple(features.shape) if isinstance(features, Tensor) else type(features).__name__
        raise KADCheckpointError(
            "KAD query-pack text_features must have shape "
            f"({len(prompts)}, {KAD512_EMBED_DIM}), got {shape}."
        )
    if not bool(torch.isfinite(features).all()):
        raise KADCheckpointError("KAD query-pack text_features contain NaN or infinity.")

    expected_preprocess = _query_pack_preprocess_metadata()
    if pack["preprocess"] != expected_preprocess:
        raise KADCheckpointError(
            "KAD query-pack preprocessing metadata does not match this adapter's "
            "512px RGB/ImageNet preprocessing."
        )
    source = pack.get("source", {})
    if not isinstance(source, Mapping):
        raise KADCheckpointError("KAD query-pack source metadata must be a mapping.")
    return {
        **dict(pack),
        "image_encoder": image_state,
        "model": model_state,
        "text_features": features.detach().float().cpu(),
        "labels": list(labels),
        "prompts": list(prompts),
        "source": dict(source),
    }


def kad512_query_pack_semantic_sha256(pack: Any) -> str:
    """Hash the validated query-pack content independently of ``torch.save`` bytes.

    PyTorch zip serialization embeds the output basename and can change container
    metadata between releases. A raw file SHA-256 is still recorded for each run,
    while this digest binds the actual tensor keys, shapes, dtypes, bytes, frozen
    queries, preprocessing contract, and stable source metadata across runtimes.
    The local checkpoint pathname is deliberately excluded; its pinned SHA-256 is
    retained.
    """

    checked = preflight_kad512_query_pack(pack)
    digest = hashlib.sha256()

    def update_bytes(label: str, payload: bytes) -> None:
        label_bytes = label.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    source = dict(checked["source"])
    source.pop("checkpoint", None)
    metadata = {
        "format": checked["format"],
        "format_version": checked["format_version"],
        "labels": checked["labels"],
        "prompts": checked["prompts"],
        "preprocess": checked["preprocess"],
        "source": source,
    }
    update_bytes(
        "metadata",
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
    )

    tensor_groups = (
        ("image_encoder", checked["image_encoder"]),
        ("model", checked["model"]),
        ("text_features", {"text_features": checked["text_features"]}),
    )
    for group_name, tensors in tensor_groups:
        for key in sorted(tensors):
            tensor = tensors[key].detach().cpu().contiguous()
            if tensor.layout != torch.strided:
                raise KADCheckpointError(
                    f"KAD query-pack tensor {group_name}.{key} is not strided."
                )
            identity = json.dumps(
                {
                    "group": group_name,
                    "key": key,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            update_bytes("tensor_identity", identity)
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            update_bytes("tensor_bytes", raw)
    return digest.hexdigest()


def _build_hf_text_encoder(
    model_id: str,
    *,
    revision: str,
    local_files_only: bool,
) -> KADTextEncoder:
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise KADCheckpointError(
            "Full KAD checkpoint loading needs transformers. Install "
            "requirements-extras.txt, or use a BERT-free KAD query pack."
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        config.output_hidden_states = True
        bert_model = AutoModel.from_config(config)
    except Exception as exc:
        raise KADCheckpointError(
            f"Could not construct the KAD text architecture from {model_id!r}: {exc}"
        ) from exc
    return KADTextEncoder(bert_model)


def _build_hf_tokenizer(
    model_id: str,
    *,
    revision: str,
    local_files_only: bool,
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise KADCheckpointError(
            "Encoding KAD prompts needs transformers. Install requirements-extras.txt."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            do_lower_case=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise KADCheckpointError(f"Could not load KAD tokenizer {model_id!r}: {exc}") from exc


def _encode_prompts(
    text_encoder: KADTextEncoder,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> Tensor:
    tokenized = tokenizer(
        list(prompts),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokens = {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized["attention_mask"].to(device),
    }
    text_encoder.eval()
    with torch.inference_mode():
        return text_encoder.encode_text(tokens).detach()


def export_kad512_query_pack(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    bert_model_id: str = KAD512_BERT_MODEL_ID,
    bert_revision: str = KAD512_BERT_REVISION,
    tokenizer_id: str | None = None,
    labels: Sequence[str] = KAD512_LABELS,
    prompts: Sequence[str] = KAD512_PROMPTS,
    max_length: int = 256,
    device: str | None = None,
    local_files_only: bool = False,
    allow_unsafe_pickle: bool = False,
    overwrite: bool = False,
    source_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Encode fixed queries once and save a lean, BERT-free KAD inference pack."""

    destination = Path(output_path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing KAD query pack: {destination}")

    checkpoint = _load_torch_mapping(
        checkpoint_path,
        description="KAD checkpoint",
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    image_encoder = KADResNet512Encoder()
    text_encoder = _build_hf_text_encoder(
        bert_model_id,
        revision=bert_revision,
        local_files_only=local_files_only,
    )
    query_decoder = KADQueryDecoder()
    load_kad512_checkpoint(
        checkpoint,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        query_decoder=query_decoder,
    )

    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    text_encoder = text_encoder.eval().to(runtime_device)
    tokenizer = _build_hf_tokenizer(
        tokenizer_id or bert_model_id,
        revision=bert_revision,
        local_files_only=local_files_only,
    )
    text_features = _encode_prompts(
        text_encoder,
        tokenizer,
        prompts,
        max_length=max_length,
        device=runtime_device,
    )
    source = {
        "kind": "official-kad-checkpoint",
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "bert_model_id": bert_model_id,
        "bert_revision": bert_revision,
        "tokenizer_id": tokenizer_id or bert_model_id,
        "max_length": int(max_length),
        **dict(source_metadata or {}),
    }
    pack = build_kad512_query_pack(
        checkpoint,
        text_features=text_features,
        labels=labels,
        prompts=prompts,
        source_metadata=source,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, destination)
    return destination


class KAD512Expert:
    """Official KAD-512 weights as a routable, zero-shot chest-X-ray expert.

    Supply exactly one of ``query_pack_path`` (recommended) or ``checkpoint_path``.
    A query pack starts without BERT or transformers.  A full checkpoint constructs
    Med-KEBERT, loads its state, and caches the configured prompt embeddings in memory.
    """

    def __init__(
        self,
        *,
        query_pack_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        name: str = "chest_kad512",
        labels: Sequence[str] = KAD512_LABELS,
        prompts: Sequence[str] = KAD512_PROMPTS,
        bert_model_id: str = KAD512_BERT_MODEL_ID,
        bert_revision: str = KAD512_BERT_REVISION,
        tokenizer_id: str | None = None,
        max_length: int = 256,
        device: str | None = None,
        amp: bool = True,
        local_files_only: bool = False,
        allow_unsafe_pickle: bool = False,
    ) -> None:
        if (query_pack_path is None) == (checkpoint_path is None):
            raise ValueError(
                "KAD512Expert needs exactly one of query_pack_path or checkpoint_path."
            )
        if len(labels) != len(prompts) or not labels:
            raise ValueError("KAD labels and prompts must be non-empty and aligned.")

        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.CHEST
        self.class_names = list(labels)
        self.prompts = list(prompts)
        self.query_pack_path = query_pack_path
        self.checkpoint_path = checkpoint_path
        self.bert_model_id = bert_model_id
        self.bert_revision = bert_revision
        self.tokenizer_id = tokenizer_id or bert_model_id
        self.max_length = int(max_length)
        self.device = device
        self.amp = bool(amp)
        self.local_files_only = bool(local_files_only)
        self.allow_unsafe_pickle = bool(allow_unsafe_pickle)

        self._image_encoder: KADResNet512Encoder | None = None
        self._query_decoder: KADQueryDecoder | None = None
        self._text_features: Tensor | None = None
        self.source_metadata: dict[str, Any] = {}

    def _load_query_pack(self) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor], Tensor]:
        pack = _load_torch_mapping(
            self.query_pack_path,
            description="KAD query pack",
            allow_unsafe_pickle=False,
        )
        checked = preflight_kad512_query_pack(pack)
        requested_default = (
            tuple(self.class_names) == KAD512_LABELS and tuple(self.prompts) == KAD512_PROMPTS
        )
        packed_identity = (
            tuple(checked["labels"]) == tuple(self.class_names)
            and tuple(checked["prompts"]) == tuple(self.prompts)
        )
        if not packed_identity and not requested_default:
            raise KADCheckpointError(
                "KAD query pack labels/prompts do not match those requested by the expert."
            )
        # A custom pack can supply its own vocabulary when the constructor used defaults.
        if not packed_identity:
            self.class_names = list(checked["labels"])
            self.prompts = list(checked["prompts"])
        self.source_metadata = checked["source"]
        return checked["image_encoder"], checked["model"], checked["text_features"]

    def _ensure_loaded(self) -> None:
        if self._image_encoder is not None:
            return
        image_encoder = KADResNet512Encoder()
        query_decoder = KADQueryDecoder()

        if self.query_pack_path is not None:
            image_state, decoder_state, text_features = self._load_query_pack()
            _validate_module_state("image_encoder", image_encoder, image_state)
            _validate_module_state("model", query_decoder, decoder_state)
            image_encoder.load_state_dict(image_state)
            query_decoder.load_state_dict(decoder_state)
        else:
            checkpoint = _load_torch_mapping(
                self.checkpoint_path,
                description="KAD checkpoint",
                allow_unsafe_pickle=self.allow_unsafe_pickle,
            )
            text_encoder = _build_hf_text_encoder(
                self.bert_model_id,
                revision=self.bert_revision,
                local_files_only=self.local_files_only,
            )
            load_kad512_checkpoint(
                checkpoint,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                query_decoder=query_decoder,
            )
            runtime_device = torch.device(
                self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            text_encoder = text_encoder.eval().to(runtime_device)
            tokenizer = _build_hf_tokenizer(
                self.tokenizer_id,
                revision=self.bert_revision,
                local_files_only=self.local_files_only,
            )
            text_features = _encode_prompts(
                text_encoder,
                tokenizer,
                self.prompts,
                max_length=self.max_length,
                device=runtime_device,
            ).cpu()
            self.source_metadata = {
                "kind": "official-kad-checkpoint",
                "checkpoint": str(Path(self.checkpoint_path).expanduser()),
                "bert_model_id": self.bert_model_id,
                "bert_revision": self.bert_revision,
                "tokenizer_id": self.tokenizer_id,
                "max_length": self.max_length,
            }
            del text_encoder

        runtime_device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = str(runtime_device)
        self._image_encoder = image_encoder.eval().to(runtime_device)
        self._query_decoder = query_decoder.eval().to(runtime_device)
        self._text_features = text_features.detach().float().to(runtime_device)

    def predict_proba(self, images: Tensor) -> Tensor:
        """Return positive-class KAD scores for an already preprocessed image batch."""

        self._ensure_loaded()
        if images.ndim != 4 or images.shape[1:] != (
            3,
            KAD512_IMAGE_SIZE,
            KAD512_IMAGE_SIZE,
        ):
            raise ValueError(
                "KAD predict_proba expects (B,3,512,512) preprocessed images, got "
                f"{tuple(images.shape)}."
            )
        runtime_device = torch.device(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp and runtime_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            image_features, _ = self._image_encoder(images.to(runtime_device))
            logits = self._query_decoder(image_features, self._text_features)
            return torch.softmax(logits.float(), dim=-1)[..., 1].cpu()

    def predict(self, scan: Scan) -> Prediction:
        image = preprocess_kad512(scan.data).unsqueeze(0)
        scores = self.predict_proba(image)[0]
        prediction = Prediction(expert=self.name, meta=scan.meta)
        prediction.class_probs = {
            label: float(score) for label, score in zip(self.class_names, scores)
        }
        # Raw KAD positive-class scores are ranking outputs, not calibrated
        # reliability estimates.  A confidence value is caller-owned and may be
        # populated only after an accepted endpoint calibration policy is applied.
        prediction.confidence = None
        prediction.meta.extra = dict(prediction.meta.extra or {})
        prediction.meta.extra["kad_prompts"] = list(self.prompts)
        prediction.meta.extra["kad_source"] = dict(self.source_metadata)
        return prediction

__all__ = [
    "KAD512Expert",
    "KAD512_BERT_MODEL_ID",
    "KAD512_BERT_REVISION",
    "KAD512_LABELS",
    "KAD512_PROMPTS",
    "KADCheckpointError",
    "KADQueryDecoder",
    "KADResNet512Encoder",
    "KADTextEncoder",
    "build_kad512_query_pack",
    "export_kad512_query_pack",
    "kad512_query_pack_semantic_sha256",
    "load_kad512_checkpoint",
    "preflight_kad512_checkpoint",
    "preflight_kad512_query_pack",
    "preprocess_kad512",
]
