"""Unit tests for oak_d_actuator.perception primitives."""
from __future__ import annotations
import numpy as np
import pytest

from oak_d_actuator.perception import (
    BlobResult,
    RED_HSV_LOW_1, RED_HSV_HIGH_1, RED_HSV_LOW_2, RED_HSV_HIGH_2, RED_MIN_AREA_PX,
    BOWL_DEPTH_MIN_MM, BOWL_DEPTH_MAX_MM, BOWL_MIN_AREA_PX,
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
