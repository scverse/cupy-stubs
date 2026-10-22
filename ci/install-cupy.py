# SPDX-License-Identifier: MPL-2.0
"""Install the `cupy` wheel for this machine’s GPU platform into the current environment.

`cupy` on PyPI is just an sdist, the wheels are in platform-specific packages.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def _cupy_requirement() -> list[str]:
    if shutil.which("nvidia-smi"):
        # CUDA libraries are hosted on NVIDIA’s index
        return ["cupy-cuda13x", "--extra-index-url=https://pypi.nvidia.com", "--index-strategy=unsafe-best-match"]
    if Path("/dev/kfd").exists():  # AMD’s compute driver
        return ["cupy-rocm-7-0"]
    sys.exit("Found neither an NVIDIA GPU (`nvidia-smi`) nor an AMD one (`/dev/kfd`), can’t choose a `cupy` wheel.")


subprocess.run(["uv", "pip", "install", "--python", sys.executable, *_cupy_requirement()], check=True)
