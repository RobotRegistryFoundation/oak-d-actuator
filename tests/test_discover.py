from unittest.mock import MagicMock, patch
from oak_d_actuator.discover import discover, DetectedOakDevice


def test_discover_returns_no_devices_when_depthai_missing():
    """If depthai isn't importable, discover() returns []."""
    import importlib
    with patch("importlib.import_module", side_effect=ImportError):
        assert discover() == []


def test_discover_returns_no_devices_when_no_hardware():
    """When depthai is importable but no device is connected, return []."""
    fake_dai = MagicMock()
    fake_dai.Device.getAllAvailableDevices.return_value = []
    with patch("importlib.import_module", return_value=fake_dai):
        assert discover() == []


def test_discover_lists_connected_devices():
    """When depthai sees devices, discover() returns DetectedOakDevice objects."""
    fake_info_a = MagicMock()
    fake_info_a.getMxId.return_value = "1844301041F5A21200"
    fake_info_a.name = "OAK-D-Lite"

    fake_info_b = MagicMock()
    fake_info_b.getMxId.return_value = "22ABCDEF12345678"
    fake_info_b.name = "OAK-D-Pro"

    fake_dai = MagicMock()
    fake_dai.Device.getAllAvailableDevices.return_value = [fake_info_a, fake_info_b]
    with patch("importlib.import_module", return_value=fake_dai):
        results = discover()

    assert len(results) == 2
    assert results[0].mxid == "1844301041F5A21200"
    assert results[0].product_family == "OAK-D-Lite"
    assert results[1].mxid == "22ABCDEF12345678"
    assert results[1].product_family == "OAK-D-Pro"


def test_discover_handles_exception_from_depthai_silently():
    """If depthai raises during enumeration, discover() returns [] instead of propagating."""
    fake_dai = MagicMock()
    fake_dai.Device.getAllAvailableDevices.side_effect = RuntimeError("usb error")
    with patch("importlib.import_module", return_value=fake_dai):
        assert discover() == []
