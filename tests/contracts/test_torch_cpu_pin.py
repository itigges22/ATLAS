"""The CPU-torch pre-install in a service Dockerfile must pin the SAME
torch version as that service's requirements.txt.

Both Dockerfiles install torch from the CPU-only index first, then run the
requirements install. When the two pins drift apart, the second pip
"upgrades" torch from PyPI and silently drags the ~8 GB nvidia/cu*
dependency stack into a CPU-only image (observed 2026-07-20: lens/v3
images at 8.29/7.91 GB instead of ~3 GB, and rebuilds failing outright on
a 43 GB host).
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SERVICES = [
    ("geometric-lens", REPO / "geometric-lens" / "Dockerfile",
     REPO / "geometric-lens" / "requirements.txt"),
    ("v3-service", REPO / "v3-service" / "Dockerfile",
     REPO / "v3-service" / "requirements.txt"),
]

DOCKERFILE_TORCH = re.compile(r"pip install [^&]*?torch==([0-9][\w.+]*)")
REQUIREMENTS_TORCH = re.compile(r"^torch==([0-9][\w.+]*)\s*$", re.MULTILINE)


@pytest.mark.parametrize("name,dockerfile,requirements",
                         SERVICES, ids=[s[0] for s in SERVICES])
def test_torch_pins_match(name, dockerfile, requirements):
    df_text = dockerfile.read_text()
    req_text = requirements.read_text()

    df_pin = DOCKERFILE_TORCH.search(df_text)
    req_pin = REQUIREMENTS_TORCH.search(req_text)
    assert df_pin, f"{name}: no torch pin found in {dockerfile}"
    assert req_pin, f"{name}: no torch pin found in {requirements}"
    assert df_pin.group(1) == req_pin.group(1), (
        f"{name}: Dockerfile pre-installs torch=={df_pin.group(1)} but "
        f"requirements.txt pins torch=={req_pin.group(1)}. The requirements "
        f"install will replace the CPU wheel with PyPI's CUDA build "
        f"(~8 GB of nvidia/cu* deps in a CPU-only image). Keep both pins "
        f"identical."
    )


@pytest.mark.parametrize("name,dockerfile,requirements",
                         SERVICES, ids=[s[0] for s in SERVICES])
def test_torch_preinstall_uses_cpu_index(name, dockerfile, requirements):
    df_text = dockerfile.read_text()
    torch_stmt = re.search(
        r"pip install[^&]*torch==[^&]*", df_text)
    assert torch_stmt, f"{name}: no torch install statement in {dockerfile}"
    assert "download.pytorch.org/whl/cpu" in torch_stmt.group(0), (
        f"{name}: the torch pre-install must use the CPU-only index "
        f"(--index-url https://download.pytorch.org/whl/cpu)."
    )
