"""MAIRA-2 — Microsoft's grounded chest-X-ray reporter, wrapped as an expert.

MAIRA-2 (`microsoft/maira-2`) is an open-weights radiology vision-language model that
reads a chest radiograph and writes a *grounded* findings section: each sentence may come
with a bounding box locating what it describes. That grounding is exactly what this
system is built around, so we wrap MAIRA-2 as an `ExpertModel` rather than as a free-text
report generator:

  * `predict(scan)` runs MAIRA-2's findings generation and stashes the raw grounded
    sequence (and the plain narrative) on the `Prediction`.
  * `findings_from_prediction(scan, pred)` parses that grounded sequence into structured
    `Finding`s — one per sentence, with a canonical label where we can match one and a
    coarse location derived from the box.

The discipline holds: MAIRA-2 decides *what* is true and *where*; the reporter still only
verbalizes the resulting findings, and the verifier checks the prose against them. The
heavy model (`transformers`, `torch`, `PIL`) loads lazily, so importing this module is
cheap and the grounded-sequence parser can be unit-tested without the weights.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from reporting.findings import Finding

# Map free-text MAIRA-2 phrases to the canonical ChestX-ray14 vocabulary, so findings
# share labels with the trained classifier (guidelines + verifier key off these). A
# phrase that matches nothing keeps a generic label and still carries its sentence.
_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Atelectasis": ("atelecta", "collapse"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "enlarged cardiac", "heart size"),
    "Effusion": ("effusion",),
    "Infiltration": ("infiltrat",),
    "Mass": ("mass",),
    "Nodule": ("nodule", "nodular"),
    "Pneumonia": ("pneumonia",),
    "Pneumothorax": ("pneumothorax",),
    "Consolidation": ("consolidation",),
    "Edema": ("edema", "oedema"),
    "Emphysema": ("emphysema",),
    "Fibrosis": ("fibrosis", "fibrotic"),
    "Pleural_Thickening": ("pleural thickening",),
    "Hernia": ("hernia",),
}

# Phrases that assert *normality* — parsed but marked not-present so nothing is invented.
_NEGATION_HINTS: tuple[str, ...] = (
    "no ", "without", "clear", "unremarkable", "normal", "no evidence", "resolved",
)


def _match_label(sentence: str) -> str | None:
    """Return the canonical label whose keywords appear in `sentence`, else None."""
    low = sentence.lower()
    for label, keywords in _LABEL_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return label
    return None


def _is_negated(sentence: str) -> bool:
    low = sentence.lower().strip()
    return any(low.startswith(h) or f" {h}" in low for h in _NEGATION_HINTS)


def _box_to_location(box: Sequence[float]) -> tuple[str | None, str | None]:
    """Coarse (laterality, zone) from a normalized [x_min,y_min,x_max,y_max] box.

    Uses the box centre. Radiographic convention: the patient's right is on the image
    left, so a centre in the left half of the image is reported as the patient's *right*.
    """
    if box is None or len(box) < 4:
        return None, None
    x_min, y_min, x_max, y_max = (float(v) for v in box[:4])
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    laterality = "right" if cx < 0.5 else "left"          # image-left == patient-right
    zone = "upper zone" if cy < 1 / 3 else ("mid zone" if cy < 2 / 3 else "lower zone")
    return laterality, zone


def parse_maira2_grounding(
    grounded: Sequence[object],
    *,
    confidence: float | None = None,
) -> tuple[str, list[Finding]]:
    """Turn MAIRA-2's grounded output into (narrative, findings) — the testable core.

    `grounded` is the sequence returned by the processor's
    `convert_output_to_plaintext_or_grounded_sequence`: each element is either a plain
    string or a `(phrase, [boxes])` tuple, where each box is a normalized
    `(x_min, y_min, x_max, y_max)`. We keep the joined text as the narrative and emit one
    `Finding` per phrase: present unless the phrase asserts normality, located from its
    first box, labelled canonically when a keyword matches (else the phrase itself).
    """
    sentences: list[str] = []
    findings: list[Finding] = []
    for element in grounded:
        if isinstance(element, str):
            phrase, boxes = element, None
        elif isinstance(element, (tuple, list)) and element:
            phrase = str(element[0])
            boxes = element[1] if len(element) > 1 else None
        else:
            continue

        phrase = phrase.strip()
        if not phrase:
            continue
        sentences.append(phrase)

        box = None
        if boxes:
            first = boxes[0] if isinstance(boxes, (list, tuple)) else boxes
            box = first
        laterality, zone = _box_to_location(box) if box is not None else (None, None)

        present = not _is_negated(phrase)
        label = _match_label(phrase) or "Finding"
        findings.append(
            Finding(
                label=label,
                probability=1.0,            # MAIRA-2 is generative; no calibrated score
                present=present,
                confidence=confidence,
                laterality=laterality,
                location=zone,
                source="maira2-grounded",
                extra={"text": phrase, "box": list(box) if box is not None else None},
            )
        )
    return " ".join(sentences), findings


class Maira2Expert:
    """`microsoft/maira-2` as a routable chest-X-ray expert producing grounded findings.

    Register it alongside the trained classifier under (XRAY, CHEST); the router returns
    both and the orchestrator runs each, so MAIRA-2's grounded findings and the
    classifier's scored findings are pooled. The model and processor load lazily on the
    first `predict` (≈7B params — GPU strongly recommended; needs `trust_remote_code`).
    """

    def __init__(
        self,
        *,
        name: str = "chest_maira2",
        model_id: str = "microsoft/maira-2",
        max_new_tokens: int = 450,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.CHEST
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.class_names: list[str] = list(_LABEL_KEYWORDS)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True
        ).eval().to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.device = device

    def _to_pil(self, scan: Scan):
        """Best-effort conversion of a (C,H,W) scan tensor to a RGB PIL image."""
        import numpy as np
        from PIL import Image

        data = scan.data
        arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        if arr.ndim == 3:                      # (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        arr = arr.astype("float32")
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
        return Image.fromarray(arr.astype("uint8")).convert("RGB")

    def predict(self, scan: Scan) -> Prediction:
        import torch

        self._ensure_loaded()
        image = self._to_pil(scan)
        processed = self._processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=None,
            technique=None,
            comparison=None,
            prior_report=None,
            return_tensors="pt",
            get_grounding=True,
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **processed, max_new_tokens=self.max_new_tokens, use_cache=True
            )
        prompt_len = processed["input_ids"].shape[-1]
        decoded = self._processor.decode(output[0][prompt_len:], skip_special_tokens=True)
        grounded = self._processor.convert_output_to_plaintext_or_grounded_sequence(decoded)

        pred = Prediction(expert=self.name, meta=scan.meta)
        pred.confidence = None  # generative model: no calibrated confidence
        pred.meta.extra = dict(pred.meta.extra or {})
        pred.meta.extra["maira2_grounded"] = grounded
        narrative, _ = parse_maira2_grounding(grounded)
        pred.meta.extra["maira2_report"] = narrative
        return pred

    def findings_from_prediction(self, scan: Scan, pred: Prediction) -> list[Finding]:
        grounded = pred.meta.extra.get("maira2_grounded") if pred.meta.extra else None
        if not grounded:
            return []
        _, findings = parse_maira2_grounding(grounded, confidence=pred.confidence)
        return findings
