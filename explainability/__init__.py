"""Explainability: why an expert decided what it did.

Grad-CAM over the shared backbone gives a faithful saliency map (the same features
every head reads), used both to localize mask-less findings and to show the doctor.
"""

from .gradcam import GradCAM

__all__ = ["GradCAM"]
