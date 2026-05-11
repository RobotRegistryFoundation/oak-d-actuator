from unittest.mock import MagicMock, patch
from oak_d_actuator.camera import Camera


def _make_fake_dai_with_calibration():
    """Build a fake depthai module with a Camera-compatible pipeline AND a
    readCalibration() that returns intrinsics + distortion."""
    fake_dai = MagicMock()
    fake_dai.Pipeline.return_value = MagicMock()

    fake_calib = MagicMock()
    # depthai returns the K matrix as a list-of-lists (3x3)
    fake_calib.getCameraIntrinsics.return_value = [
        [1000.0,    0.0, 640.0],
        [   0.0, 1000.0, 360.0],
        [   0.0,    0.0,   1.0],
    ]
    fake_calib.getDistortionCoefficients.return_value = [0.01, -0.02, 0.0, 0.0, 0.0, 0.001]

    fake_device = MagicMock()
    fake_device.readCalibration.return_value = fake_calib

    # CameraBoardSocket.CAM_A / CAM_B / CAM_C just need to be distinct sentinels
    fake_dai.CameraBoardSocket.CAM_A = "CAM_A"
    fake_dai.CameraBoardSocket.CAM_B = "CAM_B"
    fake_dai.CameraBoardSocket.CAM_C = "CAM_C"

    # Output queues — even though we don't read frames in this test, Camera's
    # __init__ calls getOutputQueue, so we need to keep it happy.
    fake_device.getOutputQueue.return_value = MagicMock()

    device_cm = MagicMock()
    device_cm.__enter__.return_value = fake_device
    fake_dai.Device.return_value = device_cm

    return fake_dai


def test_get_intrinsics_rgb_default_resolution():
    """get_intrinsics('rgb') returns fx/fy/cx/cy + distortion at the default
    resolution, sourced from factory calibration."""
    fake_dai = _make_fake_dai_with_calibration()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        intr = cam.get_intrinsics(stream="rgb")

    assert intr["fx"] == 1000.0
    assert intr["fy"] == 1000.0
    assert intr["cx"] == 640.0
    assert intr["cy"] == 360.0
    assert intr["distortion_model"] == "plumb_bob"
    # plumb_bob has 5 coefficients; the depthai readback may include more, we
    # truncate to 5.
    assert intr["distortion_coeffs"] == [0.01, -0.02, 0.0, 0.0, 0.0]
    assert intr["width"] == 1280
    assert intr["height"] == 720
    assert intr["provenance"] == "depthai factory cal"


def test_get_intrinsics_unknown_stream_raises():
    """An unsupported stream name raises ValueError with a helpful message."""
    fake_dai = _make_fake_dai_with_calibration()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        try:
            cam.get_intrinsics(stream="lidar")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "lidar" in str(e)
            assert "rgb" in str(e)  # mentions supported streams


def test_get_intrinsics_left_and_right_streams_supported():
    """get_intrinsics('left') and ('right') resolve to CameraBoardSocket.CAM_B / CAM_C."""
    fake_dai = _make_fake_dai_with_calibration()
    with patch.dict("sys.modules", {"depthai": fake_dai}):
        cam = Camera()
        intr_l = cam.get_intrinsics(stream="left")
        intr_r = cam.get_intrinsics(stream="right")

    assert intr_l["provenance"] == "depthai factory cal"
    assert intr_r["provenance"] == "depthai factory cal"
