"""Expert packs — specialized (modality, body-part) models behind one `predict` contract.

Two kinds live here now:

  * **Trained packs** — a shared backbone + task heads we train ourselves (`ChestXray`).
  * **Pretrained adapters** — released research models wrapped to the same contract with
    no local training: `TotalSegmentatorExpert` (CT organ segmentation) and
    `Maira2Expert` (grounded chest-X-ray reporting). Integration does not imply that
    their weights, outputs, or licenses have been accepted for the target setting.

`build_default_registry` assembles a registry from whichever experts you ask for, so the
pipeline can be stood up in one call.
"""

from core.enums import BodyPart, Modality
from routing import ExpertRegistry

from .chest_xray import CHESTXRAY14_LABELS, build_chest_xray_expert
from .ct_totalsegmentator import TotalSegmentatorExpert
from .kad import KAD512Expert
from .maira2 import Maira2Expert
from .msk_fracture import MSKFractureExpert
from .torchxrayvision import TorchXRayVisionExpert

__all__ = [
    "build_chest_xray_expert",
    "CHESTXRAY14_LABELS",
    "TotalSegmentatorExpert",
    "Maira2Expert",
    "KAD512Expert",
    "TorchXRayVisionExpert",
    "MSKFractureExpert",
    "build_default_registry",
]


def build_default_registry(
    *,
    chest_expert=None,
    include_xrv: bool = False,
    include_kad: bool = False,
    include_maira2: bool = False,
    include_ct: bool = False,
    include_msk: bool = False,
    xrv_kwargs: dict | None = None,
    kad_kwargs: dict | None = None,
    ct_kwargs: dict | None = None,
    msk_kwargs: dict | None = None,
) -> ExpertRegistry:
    """Assemble an `ExpertRegistry` from the experts you want active.

    `chest_expert` is your trained (or freshly built) chest-X-ray `BaseExpert`. Set
    `include_xrv` to register the pretrained TorchXRayVision engineering control under
    (XRAY, CHEST). Its scores are not accepted operating probabilities for this project;
    registering it must not be interpreted as clinical validation. Register it alongside
    `chest_expert` only when deliberately comparing their raw findings.
    `include_kad` registers research-only KAD-512 and requires either a checkpoint or
    query-pack path in `kad_kwargs`; prefer one endpoint-isolated BERT-free query pack.
    Its raw scores are not calibrated probabilities, and the pipeline's default 0.5
    threshold is not an accepted operating point. Set `include_maira2` to also
    register MAIRA-2 under (XRAY, CHEST). `include_ct` registers one TotalSegmentator
    instance under both CT niches (chest and abdomen) via `register_niche`, since one set
    of weights serves both. `include_msk` registers the pretrained YOLOv8 wrist-fracture
    detector under (XRAY, BONE). Heavy adapters are constructed only when requested.
    """
    registry = ExpertRegistry()

    if chest_expert is not None:
        registry.register(chest_expert)
    if include_xrv:
        registry.register(TorchXRayVisionExpert(**(xrv_kwargs or {})))
    if include_kad:
        registry.register(KAD512Expert(**(kad_kwargs or {})))
    if include_maira2:
        registry.register(Maira2Expert())
    if include_ct:
        ct = TotalSegmentatorExpert(**(ct_kwargs or {}))
        registry.register_niche(Modality.CT, BodyPart.CHEST, ct)
        registry.register_niche(Modality.CT, BodyPart.ABDOMEN, ct)
    if include_msk:
        registry.register(MSKFractureExpert(**(msk_kwargs or {})))

    return registry
