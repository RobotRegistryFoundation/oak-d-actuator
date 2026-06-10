"""Unit tests for the Actuator Protocol entry-point class."""
from __future__ import annotations
from unittest.mock import MagicMock
import json
from pathlib import Path

import numpy as np
import pytest

MANIFEST = Path("/tmp/ROBOT.md")


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
    outcome = a.execute(envelope={"tool_name": "perceive", "tool_args": {"query": "red_blob"}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
    assert outcome.success is True
    assert outcome.telemetry["found"] is False


def test_actuator_perceive_rejects_unknown_query():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute(envelope={"tool_name": "perceive", "tool_args": {"query": "bogus"}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
    assert outcome.success is False
    assert "unknown query" in (outcome.error_message or "")


def test_actuator_read_rgb_returns_shape_only_not_pixels():
    from oak_d_actuator.actuator import OakDActuator
    cam = MagicMock()
    cam.read_rgb.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    a = OakDActuator(camera=cam)
    outcome = a.execute(envelope={"tool_name": "read_rgb", "tool_args": {}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
    assert outcome.success is True
    assert outcome.telemetry["shape"] == [480, 640, 3]
    assert outcome.telemetry["dtype"] == "uint8"
    assert "pixels" not in outcome.telemetry


def test_actuator_unknown_tool_name():
    from oak_d_actuator.actuator import OakDActuator
    a = OakDActuator(camera=MagicMock())
    outcome = a.execute(envelope={"tool_name": "nope", "tool_args": {}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
    assert outcome.success is False
    assert "unknown capability" in (outcome.error_message or "")  # arm-driver-parity wording


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
    outcome = a.execute(envelope={"tool_name": "perceive", "tool_args": {"query": "red_blob"}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
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
    out = a.execute(envelope={"tool_name": "perceive", "tool_args": {"query": "red_blob"}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
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


def test_actuator_conforms_to_keyword_only_protocol():
    """gateway#28 regression: the dispatcher calls execute(envelope=..., ...).

    The positional 0.2.x form TypeError'd on every gateway dispatch. Pin the
    keyword-only signature and the canonical outcome fields.
    """
    import inspect
    from oak_d_actuator.actuator import OakDActuator
    sig = inspect.signature(OakDActuator.execute)
    params = list(sig.parameters.values())[1:]  # drop self
    assert [p.name for p in params] == ["envelope", "manifest_path", "tier", "config"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
    from unittest.mock import MagicMock
    a = OakDActuator(camera=MagicMock())
    out = a.execute(envelope={"tool_name": "nope", "tool_args": {}}, manifest_path=MANIFEST, tier="OBSERVE", config={})
    assert out.outcome_kind == "error"
    assert out.success is False
