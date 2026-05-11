"""oak-d-actuator — thin depth+RGB wrapper for the robot-md ecosystem.

See `Camera` for the main API surface.
See `OakDActuator` for the Actuator Protocol entry-point.
RPN-000000000003 in the RRF transparency log.
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from oak_d_actuator.camera import Camera
from oak_d_actuator.discover import discover, DetectedOakDevice
from oak_d_actuator.actuator import OakDActuator, ActuatorOutcome
from oak_d_actuator.perception import BlobResult, find_red_blob, find_bowl_top

try:
    __version__ = _pkg_version("oak-d-actuator")
except PackageNotFoundError:
    __version__ = "0.0.0+source"

__all__ = [
    "__version__",
    "Camera", "discover", "DetectedOakDevice",
    "OakDActuator", "ActuatorOutcome",
    "BlobResult", "find_red_blob", "find_bowl_top",
]
