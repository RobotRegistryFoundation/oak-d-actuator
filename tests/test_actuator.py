"""Unit tests for the Actuator Protocol entry-point class."""
from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_actuator_capabilities_include_perceive():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    assert "perceive" in a.capabilities
    assert "read_rgb" in a.capabilities
    assert "read_depth" in a.capabilities


def test_actuator_perceive_red_blob_on_zero_frame_returns_not_found():
    from oak_d_actuator.actuator import OakDActuator
    cam = MagicMock()
    cam.read_rgb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    cam.read_depth.return_value = np.zeros((480, 640), dtype=np.uint16)
    a = OakDActuator(camera=cam)
    outcome = a.execute("perceive", {"query": "red_blob"})
    assert outcome.success is True
    assert outcome.telemetry["found"] is False


def test_actuator_perceive_rejects_unknown_query():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute("perceive", {"query": "bogus"})
    assert outcome.success is False
    assert "unknown query" in (outcome.error or "")


def test_actuator_read_rgb_returns_shape_only_not_pixels():
    from oak_d_actuator.actuator import OakDActuator
    cam = MagicMock()
    cam.read_rgb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    a = OakDActuator(camera=cam)
    outcome = a.execute("read_rgb", {})
    assert outcome.success is True
    assert outcome.telemetry["shape"] == [480, 640, 3]
    assert outcome.telemetry["dtype"] == "uint8"
    assert "pixels" not in outcome.telemetry


def test_actuator_unknown_tool_name():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute("nope", {})
    assert outcome.success is False
    assert "unknown tool_name" in (outcome.error or "")


def test_actuator_perceive_red_blob_routes_to_perception_fn(monkeypatch):
    """Patch find_red_blob to verify the actuator wires rgb+depth into it."""
    from oak_d_actuator import actuator as mod
    from oak_d_actuator.perception import BlobResult
    cam = MagicMock()
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.zeros((100, 100), dtype=np.uint16)
    cam.read_rgb.return_value = rgb
    cam.read_depth.return_value = depth
    a = mod.OakDActuator(camera=cam)
    sentinel = BlobResult(found=True, centroid_px=(50, 50), pixel_count=999, provenance={"x": 1})
    monkeypatch.setattr(mod, "find_red_blob", lambda r, d: sentinel)
    outcome = a.execute("perceive", {"query": "red_blob"})
    assert outcome.success is True
    assert outcome.telemetry["found"] is True
    assert outcome.telemetry["centroid_px"] == [50, 50]
    assert outcome.telemetry["pixel_count"] == 999
    assert outcome.telemetry["provenance"] == {"x": 1}
