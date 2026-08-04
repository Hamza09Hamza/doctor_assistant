"""Standalone MAIRA-2 demo — run this in Colab, not locally.

MAIRA-2 is a 7B-parameter model (~14GB in fp16); this repo's local dev sandbox has a
6GB-VRAM GPU, which can't hold it. This script has no other dependency on this repo's
structure than experts/maira2.py itself, so it's meant to be run standalone:

    HF_TOKEN=<your token, set as an env var, never pasted into a cell>
    python scripts/demo_maira2.py

It downloads one public sample frontal chest X-ray (from the official MAIRA-2 model
card's own usage example, https://huggingface.co/microsoft/maira-2 -- IU-Xray dataset,
hosted at openi.nlm.nih.gov), runs it through Maira2Expert.predict(), and prints the
narrative report plus each parsed grounded finding (label, location, bounding box).

Licensing reminder (MSRLA, see https://huggingface.co/microsoft/maira-2/blob/main/LICENSE):
non-commercial, research-only, and explicitly forbids "a stand-alone hosted solution for
others to use." This script is for exploring output quality yourself -- it is not a
building block for a feature other people (radiologists, patients) would use.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Same frontal sample the official model card's own usage example uses.
SAMPLE_FRONTAL_URL = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"


def _load_sample_scan():
    import numpy as np
    import requests
    import torch
    from PIL import Image

    from core.enums import BodyPart, Modality
    from core.types import Scan, ScanMetadata

    response = requests.get(SAMPLE_FRONTAL_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    array = np.asarray(image).astype("float32")
    tensor = torch.from_numpy(array).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    meta = ScanMetadata(
        modality=Modality.XRAY,
        body_part=BodyPart.CHEST,
        original_shape=tuple(tensor.shape),
        source_path=SAMPLE_FRONTAL_URL,
    )
    return Scan(data=tensor, meta=meta)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print(
            "HF_TOKEN is not set. Export it before running "
            "(export HF_TOKEN=<your token>) -- MAIRA-2 is gated on Hugging Face and "
            "will fail to download without an authenticated, gate-accepted token.",
            file=sys.stderr,
        )
        return 1

    from experts.maira2 import Maira2Expert

    print(f"Fetching sample frontal CXR from {SAMPLE_FRONTAL_URL} ...")
    scan = _load_sample_scan()

    print("Loading MAIRA-2 (first run downloads ~14GB of weights; be patient) ...")
    expert = Maira2Expert()

    print("Running grounded findings generation ...")
    prediction = expert.predict(scan)
    findings = expert.findings_from_prediction(scan, prediction)

    print("\n=== Narrative report ===")
    print(prediction.meta.extra.get("maira2_report", "<empty>"))

    print(f"\n=== {len(findings)} parsed finding(s) ===")
    for finding in findings:
        box = finding.extra.get("box") if finding.extra else None
        print(
            f"- [{finding.label}] present={finding.present} "
            f"laterality={finding.laterality} location={finding.location} "
            f"box={box}\n  text: {finding.extra.get('text') if finding.extra else ''!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
