"""Unit tests for the Actuator Protocol entry-point class."""
from __future__ import annotations
from unittest.mock import MagicMock
import json

import numpy as np
import pytest


def _inv(tool_name, tool_args):
    """Keyword args exactly as robot-md-gateway passes them."""
    return {
        "envelope": {"msg_id": "t", "tool_name": tool_name, "tool_args": tool_args},
        "manifest_path": None,
        "tier": "read",
        "config": {},
    }


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
    outcome = a.execute(**_inv("perceive", {"query": "red_blob"}))
    assert outcome.success is True
    assert outcome.telemetry["found"] is False


def test_actuator_perceive_rejects_unknown_query():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute(**_inv("perceive", {"query": "bogus"}))
    assert outcome.success is False
    assert "unknown query" in (outcome.error or "")


def test_actuator_read_rgb_returns_shape_only_not_pixels():
    from oak_d_actuator.actuator import OakDActuator
    cam = MagicMock()
    cam.read_rgb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    a = OakDActuator(camera=cam)
    outcome = a.execute(**_inv("read_rgb", {}))
    assert outcome.success is True
    assert outcome.telemetry["shape"] == [480, 640, 3]
    assert outcome.telemetry["dtype"] == "uint8"
    assert "pixels" not in outcome.telemetry


def test_actuator_unknown_tool_name():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute(**_inv("nope", {}))
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
    sentinel = BlobResult(
        found=True,
        centroid_px=(50, 50),
        bbox_px=(10, 20, 60, 80),
        pixel_count=999,
        provenance={"x": 1},
    )
    monkeypatch.setattr(mod, "find_red_blob", lambda r, d: sentinel)
    outcome = a.execute(**_inv("perceive", {"query": "red_blob"}))
    assert outcome.success is True
    assert outcome.telemetry["found"] is True
    assert outcome.telemetry["centroid_px"] == [50, 50]
    assert outcome.telemetry["pixel_count"] == 999
    assert outcome.telemetry["provenance"] == {"x": 1}
    assert outcome.telemetry["bbox_px"] == [10, 20, 60, 80]


def test_jsonify_coerces_tuples_recursively():
    from oak_d_actuator.actuator import _jsonify
    assert _jsonify((1, 2)) == [1, 2]
    assert _jsonify({"a": (3, 4), "b": [5, (6, 7)]}) == {"a": [3, 4], "b": [5, [6, 7]]}
    assert _jsonify(None) is None
    assert _jsonify(42) == 42


def test_actuator_perceive_telemetry_is_json_serializable():
    """Telemetry from perceive must round-trip through json.dumps; live BlobResult
    contains tuples that must be coerced to lists."""
    from oak_d_actuator.actuator import OakDActuator
    cam = MagicMock()
    cam.read_rgb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    cam.read_depth.return_value = np.zeros((480, 640), dtype=np.uint16)
    a = OakDActuator(camera=cam)
    out = a.execute(**_inv("perceive", {"query": "red_blob"}))
    assert out.success is True
    # Round-trip through json.dumps — raises TypeError if any non-JSON-safe type leaks.
    serialized = json.dumps(out.telemetry)
    restored = json.loads(serialized)
    assert restored == out.telemetry


def test_actuator_entry_point_registration():
    from importlib.metadata import entry_points
    eps = list(entry_points(group="robot_md_gateway.actuators"))
    names = [ep.name for ep in eps]
    assert "oak-d" in names, f"oak-d entry-point missing; got {names}"
    target = next(ep for ep in eps if ep.name == "oak-d").load()
    assert target.__name__ == "OakDActuator"


def test_top_level_exports_include_actuator_and_perception():
    import oak_d_actuator
    assert "OakDActuator" in oak_d_actuator.__all__
    assert "BlobResult" in oak_d_actuator.__all__
    assert "find_red_blob" in oak_d_actuator.__all__
    assert "find_bowl_top" in oak_d_actuator.__all__


def test_matches_the_gateway_actuator_protocol():
    """robot-md-gateway#28: execute() had drifted from the gateway's call
    shape and every invoke TypeError'd. Pin both the call signature and the
    outcome fields the gateway reads to the gateway's own definitions."""
    import dataclasses
    import inspect

    gw = pytest.importorskip("robot_md_gateway.actuator")
    from oak_d_actuator.actuator import ActuatorOutcome, OakDActuator

    want = inspect.signature(gw.Actuator.execute)
    got = inspect.signature(OakDActuator.execute)
    assert list(got.parameters) == list(want.parameters)
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for name, p in got.parameters.items()
        if name != "self"
    )
    gw_fields = {f.name for f in dataclasses.fields(gw.ActuatorOutcome)}
    assert gw_fields <= {f.name for f in dataclasses.fields(ActuatorOutcome)}
