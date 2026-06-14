"""Expert packs — specialized (modality, body-part) models behind one `predict` contract.

Two kinds live here now:

  * **Trained packs** — a shared backbone + task heads we train ourselves (`ChestXray`).
  * **Pretrained adapters** — strong open models wrapped to the same contract with no
    training: `TotalSegmentatorExpert` (CT organ segmentation) and `Maira2Expert`
    (grounded chest-X-ray reporting). They plug into the router/orchestrator unchanged.

`build_default_registry` assembles a registry from whichever experts you ask for, so the
pipeline can be stood up in one call.
"""

from core.enums import BodyPart, Modality
from routing import ExpertRegistry

from .chest_xray import CHESTXRAY14_LABELS, build_chest_xray_expert
from .ct_totalsegmentator import TotalSegmentatorExpert
from .maira2 import Maira2Expert
from .torchxrayvision import TorchXRayVisionExpert

__all__ = [
    "build_chest_xray_expert",
    "CHESTXRAY14_LABELS",
    "TotalSegmentatorExpert",
    "Maira2Expert",
    "TorchXRayVisionExpert",
    "build_default_registry",
]


def build_default_registry(
    *,
    chest_expert=None,
    include_xrv: bool = False,
    include_maira2: bool = False,
    include_ct: bool = False,
    xrv_kwargs: dict | None = None,
    ct_kwargs: dict | None = None,
) -> ExpertRegistry:
    """Assemble an `ExpertRegistry` from the experts you want active.

    `chest_expert` is your trained (or freshly built) chest-X-ray `BaseExpert`. Set
    `include_xrv` to register the pretrained TorchXRayVision classifier under (XRAY, CHEST)
    — strong, calibrated weights that fire on real findings out of the box; register it
    alongside `chest_expert` and the router returns both so their findings pool. Set
    `include_maira2` to also register MAIRA-2 under (XRAY, CHEST). `include_ct` registers one
    TotalSegmentator instance under both CT niches (chest and abdomen) via `register_niche`,
    since one set of weights serves both. Heavy adapters are constructed only when requested.
    """
    registry = ExpertRegistry()

    if chest_expert is not None:
        registry.register(chest_expert)
    if include_xrv:
        registry.register(TorchXRayVisionExpert(**(xrv_kwargs or {})))
    if include_maira2:
        registry.register(Maira2Expert())
    if include_ct:
        ct = TotalSegmentatorExpert(**(ct_kwargs or {}))
        registry.register_niche(Modality.CT, BodyPart.CHEST, ct)
        registry.register_niche(Modality.CT, BodyPart.ABDOMEN, ct)

    return registry
