"""Build the real `routing.ExpertRegistry` the API serves requests against.

`scripts/smoke_system.py` builds a registry too, but only ever with a
randomly-initialized, non-pretrained wiring-test expert — there is no existing
"construct the real adapters" function to reuse.

Every expert's *class definition* lives behind lazy imports already (confirmed in
`experts/*.py` — heavy deps like `torchxrayvision`/`transformers`/`ultralytics` are
imported inside methods, not at module level), so importing `experts.kad` etc. here is
itself cheap. The real risk this module guards against is construction-time failure:
KAD has no safe default (it needs a real checkpoint/query-pack file on disk, see
`experts/kad.py::KAD512Expert.__init__`), and any expert's *module* import could still
fail in an environment where that expert's optional extra
(`requirements-extras.txt`) isn't installed, or where its own dependency stack
conflicts with another's (the vision doc's Section 11.3 concern). Each attempt is
isolated, mirroring `pipeline.py`'s own per-request failure isolation — one missing or
misconfigured expert must not stop the API from serving whichever others are usable on
this particular machine.
"""

from __future__ import annotations

import logging
import os

from routing import ExpertRegistry

logger = logging.getLogger(__name__)


def _try_register(registry: ExpertRegistry, label: str, build) -> None:
    try:
        expert = build()
    except Exception as exc:  # noqa: BLE001 — isolate one expert's failure from the rest
        logger.warning("api.registry: %s unavailable (%s: %s)", label, type(exc).__name__, exc)
        return
    if expert is None:  # builder itself decided to skip (e.g. no env var configured)
        return
    registry.register(expert)
    logger.info("api.registry: registered %s", label)


def _build_kad() -> object | None:
    from experts.kad import KAD512Expert

    query_pack_path = os.environ.get("KAD_QUERY_PACK_PATH")
    checkpoint_path = os.environ.get("KAD_CHECKPOINT_PATH")
    if query_pack_path and checkpoint_path:
        raise ValueError(
            "Both KAD_QUERY_PACK_PATH and KAD_CHECKPOINT_PATH are set; KAD512Expert "
            "requires exactly one. Fix the environment rather than guessing which to use."
        )
    if not query_pack_path and not checkpoint_path:
        logger.info(
            "api.registry: KAD skipped — set KAD_QUERY_PACK_PATH or KAD_CHECKPOINT_PATH "
            "to enable it."
        )
        return None
    return KAD512Expert(query_pack_path=query_pack_path, checkpoint_path=checkpoint_path)


def _build_torchxrayvision() -> object:
    from experts.torchxrayvision import TorchXRayVisionExpert

    return TorchXRayVisionExpert()


def _build_maira2() -> object:
    from experts.maira2 import Maira2Expert

    model_id = os.environ.get("MAIRA2_MODEL_ID", "microsoft/maira-2")
    return Maira2Expert(model_id=model_id)


def _build_msk_fracture() -> object:
    from experts.msk_fracture import MSKFractureExpert

    weights_path = os.environ.get("MSK_FRACTURE_WEIGHTS_PATH")
    return MSKFractureExpert(weights_path=weights_path)


def _build_total_segmentator() -> object:
    from core.enums import BodyPart
    from experts.ct_totalsegmentator import TotalSegmentatorExpert

    return TotalSegmentatorExpert(body_part=BodyPart.ABDOMEN)


def _build_brats() -> object | None:
    from experts.mri_brats import BraTSExpert

    bundle_root = os.environ.get("BRATS_BUNDLE_ROOT")
    if not bundle_root:
        logger.info(
            "api.registry: BraTS skipped -- set BRATS_BUNDLE_ROOT to a directory the "
            "brats_mri_segmentation bundle can be downloaded into/read from to enable it."
        )
        return None
    return BraTSExpert(bundle_root=bundle_root)


def build_default_registry() -> ExpertRegistry:
    registry = ExpertRegistry()
    _try_register(registry, "KAD-512", _build_kad)
    _try_register(registry, "TorchXRayVision", _build_torchxrayvision)
    _try_register(registry, "MAIRA-2", _build_maira2)
    _try_register(registry, "MSK fracture detector", _build_msk_fracture)
    _try_register(registry, "TotalSegmentator", _build_total_segmentator)
    _try_register(registry, "BraTS", _build_brats)
    return registry
