"""Actuator Protocol implementation for the OAK-D camera.

Exposed via the `robot_md_gateway.actuators` entry-point as name="oak-d".
Read-only sensor — exposes read_rgb / read_depth / perceive capabilities.

Implements the gateway's keyword-only Actuator Protocol
(`execute(*, envelope, manifest_path, tier, config)`) introduced with
robot-md-gateway 0.5.0a1 — the dispatcher passes the verified RCAN INVOKE
envelope whole; tool_name/tool_args live inside it. The pre-0.3 positional
form (`execute(tool_name, tool_args)`) made every gateway dispatch fail with
`TypeError: unexpected keyword argument 'envelope'`
(robot-md-gateway#28).
"""
from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any, Optional

from robot_md_gateway.actuator import ActuatorOutcome

from oak_d_actuator.camera import Camera
from oak_d_actuator.perception import find_red_blob, find_bowl_top

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
        """Dispatch a verified RCAN INVOKE envelope to the sensor capability.

        Mirrors so-arm101-actuator's migrated form: tool routing reads
        ``envelope["tool_name"]`` / ``envelope["tool_args"]``; unknown
        capabilities and sensor exceptions are both converted to structured
        error outcomes so the gateway audit chain always receives a result.
        manifest_path/tier/config are accepted per the Protocol; a read-only
        sensor currently needs none of them.
        """
        tool_name = envelope.get("tool_name")
        tool_args = envelope.get("tool_args", {}) or {}
        return self._dispatch(tool_name, tool_args)

    def _dispatch(self, tool_name: Any, tool_args: dict[str, Any]) -> ActuatorOutcome:
        if tool_name == "read_rgb":
            try:
                rgb = self._ensure_camera().read_rgb()
            except Exception as exc:
                return ActuatorOutcome(
                    success=False, outcome_kind="error",
                    error_message=f"read_rgb_error: {exc!r}",
                )
            return ActuatorOutcome(
                success=True, outcome_kind="executed",
                telemetry={"shape": list(rgb.shape), "dtype": str(rgb.dtype)},
            )
        if tool_name == "read_depth":
            try:
                depth = self._ensure_camera().read_depth()
            except Exception as exc:
                return ActuatorOutcome(
                    success=False, outcome_kind="error",
                    error_message=f"read_depth_error: {exc!r}",
                )
            return ActuatorOutcome(
                success=True, outcome_kind="executed",
                telemetry={"shape": list(depth.shape), "dtype": str(depth.dtype)},
            )
        if tool_name == "perceive":
            query = tool_args.get("query")
            if query not in _PERCEIVE_FNS:
                return ActuatorOutcome(
                    success=False, outcome_kind="error",
                    error_message=f"unknown query: {query!r}",
                )
            try:
                cam = self._ensure_camera()
                rgb = cam.read_rgb()
                depth = cam.read_depth()
                # Re-resolve through module so monkeypatching in tests works.
                from oak_d_actuator import actuator as _self_mod
                fn = {"red_blob": _self_mod.find_red_blob, "bowl_top": _self_mod.find_bowl_top}[query]
                result = fn(rgb, depth)
            except Exception as exc:
                return ActuatorOutcome(
                    success=False, outcome_kind="error",
                    error_message=f"perception_error: {exc!r}",
                )
            return ActuatorOutcome(
                success=True, outcome_kind="executed",
                telemetry=_jsonify(dataclasses.asdict(result)),
            )
        return ActuatorOutcome(
            success=False, outcome_kind="error",
            error_message=f"unknown capability: {tool_name!r}",
        )
