"""Actuator Protocol implementation for the OAK-D camera.

Exposed via the `robot_md_gateway.actuators` entry-point as name="oak-d".
Read-only sensor — exposes read_rgb / read_depth / perceive capabilities.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from oak_d_actuator.camera import Camera
from oak_d_actuator.perception import find_red_blob, find_bowl_top


@dataclass
class ActuatorOutcome:
    """Same fields as ``robot_md_gateway.actuator.ActuatorOutcome``.

    Kept local so a camera package does not pull in the gateway, but the
    gateway reads these attributes by name (outcome_kind, error_message,
    telemetry_path), so the shape must match it field for field.
    """

    success: bool
    outcome_kind: str  # "executed" | "error"
    telemetry: dict = field(default_factory=dict)
    error_message: Optional[str] = None
    telemetry_path: Optional[Path] = None

    @property
    def error(self) -> Optional[str]:
        return self.error_message


def _ok(telemetry: dict) -> ActuatorOutcome:
    return ActuatorOutcome(success=True, outcome_kind="executed", telemetry=telemetry)


def _fail(message: str) -> ActuatorOutcome:
    return ActuatorOutcome(success=False, outcome_kind="error", error_message=message)


_PERCEIVE_FNS = {"red_blob": find_red_blob, "bowl_top": find_bowl_top}


def _jsonify(obj: Any) -> Any:
    """Coerce tuples → lists recursively so telemetry is JSON-serializable.

    dataclasses.asdict preserves tuple types; the gateway emits telemetry over
    JSON where tuples must become lists. Normalize here once, at the edge.
    """
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


class OakDActuator:
    name = "oak-d"
    description = "OAK-D depth+RGB sensor as Actuator Protocol read-only sensor. RPN-000000000003."
    capabilities = ("read_rgb", "read_depth", "perceive")
    config_schema: dict = {}

    def __init__(self, camera: Optional[Camera] = None) -> None:
        self._camera = camera

    def _ensure_camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera()
        return self._camera

    def execute(
        self,
        *,
        envelope: dict,
        manifest_path: Path,
        tier: str,
        config: dict,
    ) -> ActuatorOutcome:
        """The gateway's Actuator Protocol call (robot-md-gateway#28).

        This used to take ``(tool_name, tool_args)``, which the gateway never
        passes, so every invoke against oak-d failed with a TypeError before
        reaching the camera. Every capability here is a read, so ``tier`` and
        ``manifest_path`` need no checks of their own.
        """
        return self._dispatch(envelope.get("tool_name"), envelope.get("tool_args") or {})

    def _dispatch(self, tool_name: Optional[str], tool_args: dict[str, Any]) -> ActuatorOutcome:
        if tool_name == "read_rgb":
            try:
                rgb = self._ensure_camera().read_rgb()
            except Exception as exc:
                return _fail(f"read_rgb_error: {exc!r}")
            return _ok({"shape": list(rgb.shape), "dtype": str(rgb.dtype)})
        if tool_name == "read_depth":
            try:
                depth = self._ensure_camera().read_depth()
            except Exception as exc:
                return _fail(f"read_depth_error: {exc!r}")
            return _ok({"shape": list(depth.shape), "dtype": str(depth.dtype)})
        if tool_name == "perceive":
            query = tool_args.get("query")
            if query not in _PERCEIVE_FNS:
                return _fail(f"unknown query: {query!r}")
            try:
                cam = self._ensure_camera()
                rgb = cam.read_rgb()
                depth = cam.read_depth()
                # Re-resolve through module so monkeypatching in tests works.
                from oak_d_actuator import actuator as _self_mod
                fn = {"red_blob": _self_mod.find_red_blob, "bowl_top": _self_mod.find_bowl_top}[query]
                result = fn(rgb, depth)
            except Exception as exc:
                return _fail(f"perception_error: {exc!r}")
            return _ok(_jsonify(dataclasses.asdict(result)))
        return _fail(f"unknown tool_name: {tool_name!r}")
