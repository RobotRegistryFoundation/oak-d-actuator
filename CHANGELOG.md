# Changelog

## [0.2.2] — 2026-10-02

### Fixed
- `OakDActuator.execute()` now takes the gateway's Actuator Protocol call,
  `execute(*, envelope, manifest_path, tier, config)`. It still took
  `(tool_name, tool_args)`, so every invoke through robot-md-gateway failed
  with `TypeError: unexpected keyword argument 'envelope'` before reaching
  the camera (robot-md-gateway#28). `ActuatorOutcome` gains the fields the
  gateway reads (`outcome_kind`, `error_message`, `telemetry_path`); `.error`
  remains as a read-only alias of `error_message`. A test pins both shapes to
  the gateway's own definitions when the gateway is installed.

## [0.2.1] — 2026-05-11

### Fixed
- Pin `depthai>=2.30,<3`. `camera.py` is written against the depthai 2.x
  Pipeline API (`dai.node.XLinkOut`, `dai.node.ColorCamera`,
  `dai.node.MonoCamera`, `dai.node.StereoDepth`). depthai 3.x removed
  these constructors, so 0.2.0's `depthai>=3.0` declaration installed
  cleanly but `Camera()` raised
  `AttributeError: module 'depthai.node' has no attribute 'XLinkOut'`
  on first use.

## [0.2.0] — 2026-05-11

Initial Actuator Protocol implementation. `perceive` capability with
`find_red_blob` (HSV mask + morphology + contour) and `find_bowl_top`
(depth-band contour). Entry point registered for
`robot_md_gateway.actuators`.
