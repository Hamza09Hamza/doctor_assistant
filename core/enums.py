"""Controlled vocabularies shared across the whole pipeline.

These enums are the common language the router, the experts, and the registry
all speak. Keeping them in one place means adding a new modality or body part is
a single edit, not a scavenger hunt through string literals.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Imaging modality. `str` mixin so values serialize cleanly to YAML/JSON."""

    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    FUNDUS = "fundus"
    OCT = "oct"
    UNKNOWN = "unknown"


class BodyPart(str, Enum):
    BRAIN = "brain"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BONE = "bone"
    SPINE = "spine"
    BREAST = "breast"
    HEART = "heart"
    EYE = "eye"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """What a given head produces. Drives how its output is decoded."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    CONFIDENCE = "confidence"


# Modalities that are inherently volumetric (3D). The IO and model layers use
# this to decide spatial_dims (2 vs 3) when it isn't given explicitly.
VOLUMETRIC_MODALITIES: frozenset[Modality] = frozenset({Modality.CT, Modality.MRI})
