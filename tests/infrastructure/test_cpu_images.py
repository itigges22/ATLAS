"""Dependency-boundary checks for CPU-only supporting services."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_lens_image_installs_torch_from_cpu_only_index():
    dockerfile = (ROOT / "geometric-lens" / "Dockerfile").read_text()
    requirements = (ROOT / "geometric-lens" / "requirements.txt").read_text()
    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.12.1" in dockerfile
    assert "--extra-index-url" not in requirements
    assert 'xgboost-cpu==3.2.0; sys_platform == "linux"' in requirements
