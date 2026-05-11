"""Perception primitives for pick-and-place verification.

Pure functions: take (rgb, depth) numpy arrays, return BlobResult.
Deterministic — same input always yields same output.
No model loading, no random sampling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import cv2  # type: ignore[import]
import numpy as np


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


def find_red_blob(rgb: np.ndarray, depth: np.ndarray) -> BlobResult:
    """Find the largest red blob in the RGB frame; report depth from masked region.

    Args:
        rgb: (H, W, 3) uint8 BGR (depthai/OpenCV convention).
        depth: (H, W) uint16 millimeters. Zero pixels = invalid/unknown depth.

    Returns:
        BlobResult.found=True iff a contiguous red region of >= RED_MIN_AREA_PX exists.
        centroid_depth_mm is the median of non-zero depth values inside the blob's mask;
        None if no valid depth pixels overlap the blob.
    """
    provenance = {
        "hsv_low_1": list(RED_HSV_LOW_1),
        "hsv_high_1": list(RED_HSV_HIGH_1),
        "hsv_low_2": list(RED_HSV_LOW_2),
        "hsv_high_2": list(RED_HSV_HIGH_2),
        "min_area_px": RED_MIN_AREA_PX,
    }
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(RED_HSV_LOW_1), np.array(RED_HSV_HIGH_1))
    mask2 = cv2.inRange(hsv, np.array(RED_HSV_LOW_2), np.array(RED_HSV_HIGH_2))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BlobResult(found=False, provenance=provenance)
    largest = max(contours, key=cv2.contourArea)
    # Gate uses cv2.contourArea (Green's-theorem polygon area). The fill-mask
    # is not yet drawn at this point, so this is the cheapest correct gate.
    area = int(cv2.contourArea(largest))
    if area < RED_MIN_AREA_PX:
        # Sub-threshold branch returns contour area so callers can see how close
        # the largest candidate got to RED_MIN_AREA_PX.
        return BlobResult(found=False, pixel_count=area, provenance=provenance)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return BlobResult(found=False, pixel_count=area, provenance=provenance)
    cu = int(moments["m10"] / moments["m00"])
    cv_centroid = int(moments["m01"] / moments["m00"])
    x, y, w, h = cv2.boundingRect(largest)
    bbox = (int(x), int(y), int(x + w), int(y + h))

    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [largest], -1, 255, thickness=cv2.FILLED)
    masked_depth = depth[blob_mask > 0]
    valid = masked_depth[masked_depth > 0]
    median_depth = int(np.median(valid)) if valid.size > 0 else None

    # Spec field doc says "number of mask pixels" — count lit pixels in the
    # filled raster mask, not the Green's-theorem polygon area.
    mask_pixel_count = int(cv2.countNonZero(blob_mask))
    return BlobResult(
        found=True,
        centroid_px=(cu, cv_centroid),
        centroid_depth_mm=median_depth,
        bbox_px=bbox,
        pixel_count=mask_pixel_count,
        provenance=provenance,
    )


def find_bowl_top(rgb: np.ndarray, depth: np.ndarray) -> BlobResult:
    """Find the largest depth region in the bowl-top band; ignore RGB.

    The bowl is segmented purely by depth — assumes a bird's-eye OAK-D mount
    pointed at the workspace; the bowl rim sits in [BOWL_DEPTH_MIN_MM,
    BOWL_DEPTH_MAX_MM] from the camera.
    """
    provenance = {
        "bowl_depth_min_mm": BOWL_DEPTH_MIN_MM,
        "bowl_depth_max_mm": BOWL_DEPTH_MAX_MM,
        "min_area_px": BOWL_MIN_AREA_PX,
    }
    band = ((depth >= BOWL_DEPTH_MIN_MM) & (depth <= BOWL_DEPTH_MAX_MM)).astype(np.uint8) * 255
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BlobResult(found=False, provenance=provenance)
    largest = max(contours, key=cv2.contourArea)
    # Gate uses cv2.contourArea (Green's-theorem polygon area).
    area = int(cv2.contourArea(largest))
    if area < BOWL_MIN_AREA_PX:
        # Sub-threshold branch returns contour area so callers see how close
        # the largest candidate got to BOWL_MIN_AREA_PX.
        return BlobResult(found=False, pixel_count=area, provenance=provenance)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return BlobResult(found=False, pixel_count=area, provenance=provenance)
    cu = int(moments["m10"] / moments["m00"])
    cv_centroid = int(moments["m01"] / moments["m00"])
    x, y, w, h = cv2.boundingRect(largest)
    bbox = (int(x), int(y), int(x + w), int(y + h))

    blob_mask = np.zeros_like(band)
    cv2.drawContours(blob_mask, [largest], -1, 255, thickness=cv2.FILLED)
    masked_depth = depth[blob_mask > 0]
    valid = masked_depth[masked_depth > 0]
    median_depth = int(np.median(valid)) if valid.size > 0 else None

    # Spec field doc says "number of mask pixels" — count lit pixels, not polygon area.
    mask_pixel_count = int(cv2.countNonZero(blob_mask))
    return BlobResult(
        found=True,
        centroid_px=(cu, cv_centroid),
        centroid_depth_mm=median_depth,
        bbox_px=bbox,
        pixel_count=mask_pixel_count,
        provenance=provenance,
    )
