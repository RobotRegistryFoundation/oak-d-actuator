"""Unit tests for oak_d_actuator.perception primitives."""
from __future__ import annotations
import cv2  # type: ignore[import]
import numpy as np
import pytest

from oak_d_actuator.perception import (
    BlobResult,
    RED_HSV_LOW_1, RED_HSV_HIGH_1, RED_HSV_LOW_2, RED_HSV_HIGH_2, RED_MIN_AREA_PX,
    BOWL_DEPTH_MIN_MM, BOWL_DEPTH_MAX_MM, BOWL_MIN_AREA_PX,
    find_red_blob,
    find_bowl_top,
)


def test_blob_result_defaults_to_not_found():
    r = BlobResult(found=False)
    assert r.found is False
    assert r.centroid_px is None
    assert r.centroid_depth_mm is None
    assert r.bbox_px is None
    assert r.pixel_count == 0
    assert r.provenance == {}


def test_red_thresholds_are_what_the_spec_says():
    assert RED_HSV_LOW_1 == (0, 120, 80)
    assert RED_HSV_HIGH_1 == (10, 255, 255)
    assert RED_HSV_LOW_2 == (170, 120, 80)
    assert RED_HSV_HIGH_2 == (180, 255, 255)
    assert RED_MIN_AREA_PX == 150


def test_bowl_thresholds_are_what_the_spec_says():
    assert BOWL_DEPTH_MIN_MM == 200
    assert BOWL_DEPTH_MAX_MM == 350
    assert BOWL_MIN_AREA_PX == 3000


def _zeros_rgb(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _zeros_depth(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint16)


def _put_red_patch(rgb: np.ndarray, u: int = 300, v: int = 200, size: int = 30) -> np.ndarray:
    # BGR red (depthai/OpenCV convention)
    rgb[v:v + size, u:u + size] = [0, 0, 255]
    return rgb


def test_red_blob_not_found_on_zero_frame():
    r = find_red_blob(_zeros_rgb(), _zeros_depth())
    assert r.found is False
    assert r.pixel_count == 0


def test_red_blob_found_on_red_patch():
    rgb = _put_red_patch(_zeros_rgb())
    depth = _zeros_depth()
    depth[200:230, 300:330] = 327  # mm
    r = find_red_blob(rgb, depth)
    assert r.found is True
    assert r.pixel_count >= 150
    cu, cv = r.centroid_px
    assert 295 <= cu <= 335
    assert 195 <= cv <= 235
    assert r.centroid_depth_mm == 327


def test_red_blob_returns_largest_when_two_regions():
    rgb = _zeros_rgb()
    rgb[100:115, 100:115] = [0, 0, 255]   # small (~225 px)
    rgb[200:240, 200:240] = [0, 0, 255]   # large (~1600 px)
    r = find_red_blob(rgb, _zeros_depth())
    assert r.found is True
    cu, cv = r.centroid_px
    assert 195 <= cu <= 245
    assert 195 <= cv <= 245


def test_red_blob_skips_below_min_area():
    rgb = _zeros_rgb()
    # 10x10 = 100 pixels, below 150 threshold
    rgb[100:110, 100:110] = [0, 0, 255]
    r = find_red_blob(rgb, _zeros_depth())
    assert r.found is False


def test_red_blob_bbox_contains_centroid():
    rgb = _put_red_patch(_zeros_rgb())
    r = find_red_blob(rgb, _zeros_depth())
    u0, v0, u1, v1 = r.bbox_px
    cu, cv = r.centroid_px
    assert u0 <= cu <= u1
    assert v0 <= cv <= v1


def test_red_blob_provenance_records_thresholds():
    rgb = _put_red_patch(_zeros_rgb())
    r = find_red_blob(rgb, _zeros_depth())
    p = r.provenance
    assert p["min_area_px"] == 150
    assert p["hsv_low_1"] == [0, 120, 80]
    assert p["hsv_high_1"] == [10, 255, 255]
    assert p["hsv_low_2"] == [170, 120, 80]
    assert p["hsv_high_2"] == [180, 255, 255]


def test_red_blob_depth_none_when_all_zero_in_mask():
    # Red patch present but depth frame is all zeros → median depth is None
    rgb = _put_red_patch(_zeros_rgb())
    r = find_red_blob(rgb, _zeros_depth())
    assert r.found is True
    assert r.centroid_depth_mm is None


def test_red_blob_found_on_high_hue_red():
    """Exercises mask2 (hue range [170, 180]) so a transposed mask1/mask2 bug surfaces."""
    hsv_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv_frame[200:230, 300:330] = [175, 200, 200]  # H=175 (in mask2 range), S=200, V=200
    rgb = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    r = find_red_blob(rgb, _zeros_depth())
    assert r.found is True


def test_bowl_top_not_found_on_zero_depth():
    r = find_bowl_top(_zeros_rgb(), _zeros_depth())
    assert r.found is False


def test_bowl_top_found_on_depth_band():
    depth = _zeros_depth()
    # 70x80 = 5600 px in band — above the 3000 px threshold
    depth[100:170, 100:180] = 280  # within 200-350 band
    r = find_bowl_top(_zeros_rgb(), depth)
    assert r.found is True
    cu, cv = r.centroid_px
    assert 135 <= cu <= 145
    assert 130 <= cv <= 140
    assert r.centroid_depth_mm == 280


def test_bowl_top_excludes_depths_outside_band():
    depth = _zeros_depth()
    depth[100:170, 100:180] = 100  # too close (< 200mm)
    r = find_bowl_top(_zeros_rgb(), depth)
    assert r.found is False


def test_bowl_top_skips_below_min_area():
    depth = _zeros_depth()
    # 30x30 = 900 px, below 3000 threshold
    depth[100:130, 100:130] = 280
    r = find_bowl_top(_zeros_rgb(), depth)
    assert r.found is False
