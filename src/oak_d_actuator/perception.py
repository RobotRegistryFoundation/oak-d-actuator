"""Perception primitives for pick-and-place verification.

Pure functions: take (rgb, depth) numpy arrays, return BlobResult.
Deterministic — same input always yields same output.
No model loading, no random sampling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Tuned for indoor ambient lighting + Lego red on a matte table.
# Two HSV ranges because hue 0 wraps; red lies in both [0, 10] and [170, 180].
RED_HSV_LOW_1 = (0, 120, 80)
RED_HSV_HIGH_1 = (10, 255, 255)
RED_HSV_LOW_2 = (170, 120, 80)
RED_HSV_HIGH_2 = (180, 255, 255)
RED_MIN_AREA_PX = 150

# Bowl depth band assumes bird's-eye OAK-D mount ~30cm above table.
# Below 20cm = above the bowl rim; below 35cm = below the bowl base.
BOWL_DEPTH_MIN_MM = 200
BOWL_DEPTH_MAX_MM = 350
BOWL_MIN_AREA_PX = 3000

MORPHOLOGY_KERNEL_SIZE = 3


@dataclass
class BlobResult:
    found: bool
    centroid_px: Optional[tuple[int, int]] = None
    centroid_depth_mm: Optional[int] = None
    bbox_px: Optional[tuple[int, int, int, int]] = None
    pixel_count: int = 0
    provenance: dict = field(default_factory=dict)
