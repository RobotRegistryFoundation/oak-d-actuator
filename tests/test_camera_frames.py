from unittest.mock import MagicMock, patch
import numpy as np
from oak_d_actuator.camera import Camera


def _make_fake_dai(rgb_shape=(720, 1280, 3), depth_shape=(720, 1280)):
    """Build a MagicMock depthai module that yields one RGB + one depth frame."""
    fake_dai = MagicMock()

    # Pipeline + nodes are no-op MagicMocks
    fake_dai.Pipeline.return_value = MagicMock()

    # The Device context manager returns an object whose getOutputQueue
    # returns queues that yield the expected frame shapes.
    fake_device = MagicMock()

    fake_rgb_frame = MagicMock()
    fake_rgb_frame.getCvFrame.return_value = np.zeros(rgb_shape, dtype=np.uint8)
    fake_depth_frame = MagicMock()
    fake_depth_frame.getFrame.return_value = np.zeros(depth_shape, dtype=np.uint16)

    fake_rgb_q = MagicMock()
    fake_rgb_q.get.return_value = fake_rgb_frame
    fake_depth_q = MagicMock()
    fake_depth_q.get.return_value = fake_depth_frame

    fake_device.getOutputQueue.side_effect = lambda name, **kw: {
        "rgb": fake_rgb_q,
        "depth": fake_depth_q,
    }[name]

    # Device(pipeline) returns a context-manager-style object whose __enter__
    # produces fake_device. The Camera class uses __enter__() directly.
    device_cm = MagicMock()
    device_cm.__enter__.return_value = fake_device
    fake_dai.Device.return_value = device_cm

    return fake_dai


def test_camera_read_rgb_returns_numpy_bgr():
    """read_rgb() returns a numpy array shaped (H, W, 3) dtype uint8."""
    fake_dai = _make_fake_dai()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        rgb = cam.read_rgb()

    assert rgb.shape == (720, 1280, 3)
    assert rgb.dtype == np.uint8


def test_camera_read_depth_returns_uint16_mm():
    """read_depth() returns a numpy array shaped (H, W) dtype uint16 (millimeters)."""
    fake_dai = _make_fake_dai()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        depth = cam.read_depth()

    assert depth.shape == (720, 1280)
    assert depth.dtype == np.uint16


def test_camera_release_is_idempotent():
    """release() can be called multiple times without raising."""
    fake_dai = _make_fake_dai()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        cam.release()
        cam.release()  # idempotent — no exception
