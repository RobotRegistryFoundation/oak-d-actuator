# oak-d-actuator

> **Thin depth+RGB wrapper for the robot-md ecosystem. RPN-000000000003.**

A Python package mirroring `so-arm101-actuator`'s API shape for OAK-D cameras (Luxonis). Lazy-imports depthai so the package installs on machines without the SDK; throws clearly when you actually try to open a device without it.

## Install

```bash
pip install oak-d-actuator
```

Or via `robot-md init` auto-detect, which pip-installs this package when an OAK device is detected on the host.

## Usage

```python
from oak_d_actuator import Camera

cam = Camera()
rgb = cam.read_rgb()            # (H, W, 3) uint8 BGR
depth = cam.read_depth()        # (H, W) uint16 millimeters
intr = cam.get_intrinsics(stream="rgb")
# {'fx': ..., 'fy': ..., 'cx': ..., 'cy': ...,
#  'distortion_model': 'plumb_bob',
#  'distortion_coeffs': [...],
#  'width': 1280, 'height': 720,
#  'provenance': 'depthai factory cal'}
cam.release()
```

## Discovery

```python
from oak_d_actuator import discover

for dev in discover():
    print(dev.mxid, dev.product_family)
```

## Provenance

- RPN-000000000003 in the [RRF transparency log](https://robotregistryfoundation.org/r/RPN-000000000003).
- Apache-2.0.
- Source: https://github.com/RobotRegistryFoundation/oak-d-actuator
- Ecosystem: https://robotmd.dev
