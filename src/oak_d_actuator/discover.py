"""OAK device enumeration. Calls depthai.Device.getAllAvailableDevices()."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class DetectedOakDevice:
    mxid: str            # 18-char serial
    product_family: str  # "OAK-D", "OAK-D-Lite", "OAK-1", etc.


def discover() -> list[DetectedOakDevice]:
    """Return all connected OAK devices.

    Returns [] silently if depthai isn't installed, or if no device is connected,
    or if enumeration raises. Never raises.
    """
    try:
        dai = importlib.import_module("depthai")
    except ImportError:
        return []
    try:
        infos = dai.Device.getAllAvailableDevices()
    except Exception:
        return []
    out: list[DetectedOakDevice] = []
    for info in infos:
        try:
            mxid = info.getMxId()
        except Exception:
            mxid = "unknown"
        name = getattr(info, "name", "OAK")
        out.append(DetectedOakDevice(mxid=str(mxid), product_family=str(name)))
    return out
