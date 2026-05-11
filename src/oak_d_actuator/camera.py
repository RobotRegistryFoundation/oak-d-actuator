"""High-level OAK-D camera wrapper.

Mirrors so-arm101-actuator's API shape: construct, call read_*, release.
Lazy-imports depthai so import works on machines without the SDK installed.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


class Camera:
    """A single OAK-D device, opened on construction and ready to grab frames."""

    def __init__(self, *, rgb_width: int = 1280, rgb_height: int = 720) -> None:
        dai = importlib.import_module("depthai")
        # Pipeline construction: RGB camera + StereoDepth → output queues
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(rgb_width, rgb_height)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        mono_l.out.link(depth.left)
        mono_r.out.link(depth.right)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        depth.depth.link(xout_depth.input)

        self._dai = dai
        self._pipeline = pipeline
        self._device = dai.Device(pipeline).__enter__()
        self._q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self._q_depth = self._device.getOutputQueue("depth", maxSize=4, blocking=False)
        self._rgb_width = rgb_width
        self._rgb_height = rgb_height
        self._closed = False

    def read_rgb(self) -> np.ndarray:
        """Grab the latest RGB frame as (H, W, 3) uint8 BGR."""
        frame = self._q_rgb.get()
        return frame.getCvFrame()

    def read_depth(self) -> np.ndarray:
        """Grab the latest depth frame as (H, W) uint16 millimeters."""
        frame = self._q_depth.get()
        return frame.getFrame()

    def release(self) -> None:
        """Close the device. Idempotent."""
        if self._closed:
            return
        try:
            self._device.__exit__(None, None, None)
        except Exception:
            pass
        self._closed = True
