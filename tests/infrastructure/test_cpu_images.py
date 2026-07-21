"""Dependency-boundary checks for CPU-only supporting services."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_lens_image_installs_torch_from_cpu_only_index():
    dockerfile = (ROOT / "geometric-lens" / "Dockerfile").read_text()
    requirements = (ROOT / "geometric-lens" / "requirements.txt").read_text()
    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    # The pre-install must pin the SAME version requirements pins — a
    # hardcoded version here previously let the two drift (Dockerfile
    # 2.12.1 vs requirements 2.13.0), which made the requirements install
    # replace the CPU wheel with PyPI's CUDA build (~8 GB of nvidia/cu*
    # deps in a CPU-only image). tests/contracts/test_torch_cpu_pin.py
    # guards both services; this keeps the lens file self-consistent.
    req_torch = re.search(r"^torch==(\S+)\s*$", requirements, re.MULTILINE)
    assert req_torch, "no torch pin in geometric-lens/requirements.txt"
    assert f"torch=={req_torch.group(1)}" in dockerfile
    assert "--extra-index-url" not in requirements
    assert 'xgboost-cpu==3.2.0; sys_platform == "linux"' in requirements
