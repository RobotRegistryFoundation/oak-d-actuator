"""oak-d-actuator — thin depth+RGB wrapper for the robot-md ecosystem.

See `Camera` for the main API surface.
RPN-000000000003 in the RRF transparency log.
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("oak-d-actuator")
except PackageNotFoundError:
    __version__ = "0.0.0+source"

__all__ = ["__version__"]
