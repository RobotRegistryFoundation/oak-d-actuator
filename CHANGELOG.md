# Changelog

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
