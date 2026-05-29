"""Expert packs — specialized (modality, body-part) models, each a `BaseExpert`.

One pack per clinical domain (ChestXray, NeuroMRI, BoneXray, ...). They share the same
backbone/head machinery and the same `Prediction` contract, so the router and reporter
treat every pack uniformly. Start narrow (one strong pack), unify into a platform later.
"""

from .chest_xray import CHESTXRAY14_LABELS, build_chest_xray_expert

__all__ = ["build_chest_xray_expert", "CHESTXRAY14_LABELS"]
