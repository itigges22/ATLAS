"""Verify docker-compose.yml + .env.example are internally consistent.

After env substitution, the resolved compose config must satisfy several
invariants that the rest of the stack depends on:

  - vllm-gen is on port 8000 serving qwen3.5-9b
  - vllm-embed is on port 8001 with --runner pooling --convert embed
  - geometric-lens listens on 31144 (matching its Dockerfile)
  - atlas-proxy reads LLAMA_GEN_URL pointing at vllm-gen:8000

Past iterations of this port have had silent mismatches: the Lens
Dockerfile listening on 8099 while compose expected 31144 (stage 13),
or LLAMA_GEN_URL defaulting to llama-gen-service:8000 in K8s while
compose used vllm-gen:8000 (stage 12). This test catches that class
of bug at PR review time.
"""

import re
import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(text: str, defaults: dict) -> str:
    """Mimic docker-compose's ${VAR:-default} substitution."""
    def sub(m):
        expr = m.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return defaults.get(var.strip(), default)
        return defaults.get(expr.strip(), "")
    return re.sub(r"\$\{([^}]+)\}", sub, text)


def _read_env_example() -> dict:
    out = {}
    with open(PROJECT_ROOT / ".env.example") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@pytest.fixture(scope="module")
def compose():
    """Resolved docker-compose config using .env.example defaults."""
    raw = (PROJECT_ROOT / "docker-compose.yml").read_text()
    defaults = _read_env_example()
    resolved = _resolve(raw, defaults)
    return yaml.safe_load(resolved)


def test_required_services_exist(compose):
    services = compose["services"]
    for required in ("vllm-gen", "vllm-embed", "geometric-lens",
                     "v3-service", "sandbox", "atlas-proxy"):
        assert required in services, f"missing service: {required}"


def test_vllm_gen_command_shape(compose):
    cmd = compose["services"]["vllm-gen"]["command"]
    assert "qwen3.5-9b" in cmd, "gen instance must serve qwen3.5-9b"
    assert "--reasoning-parser qwen3" in cmd, "gen needs Qwen3 reasoning parser"
    assert "--enable-prefix-caching" in cmd, "gen should have prefix caching on"
    assert "--port 8000" in cmd
    assert "--task embed" not in cmd, "gen instance must NOT be in embed mode"
    assert "--runner pooling" not in cmd, "gen instance must NOT be in pooling mode"


def test_vllm_embed_command_shape(compose):
    cmd = compose["services"]["vllm-embed"]["command"]
    assert "qwen3.5-9b-embed" in cmd, "embed instance must use the embed served-name"
    assert "--runner pooling" in cmd, "embed must use the current --runner pooling API"
    assert "--convert embed" in cmd, "embed must use --convert embed"
    assert "--task embed" not in cmd, "embed must use the new --runner/--convert API, not deprecated --task"
    assert "--port 8001" in cmd


def test_geometric_lens_healthcheck_port(compose):
    """Lens container listens on 31144 (per its Dockerfile). The
    healthcheck inside the container must hit that same port."""
    hc = compose["services"]["geometric-lens"]["healthcheck"]["test"]
    assert "31144" in str(hc), f"Lens healthcheck does not target 31144: {hc}"


def test_lens_deps_include_both_vllm_instances(compose):
    deps = compose["services"]["geometric-lens"]["depends_on"]
    # depends_on can be a list or a dict-with-conditions
    dep_names = list(deps) if isinstance(deps, (list, dict)) else []
    assert "vllm-gen" in dep_names, "Lens must wait for gen instance"
    assert "vllm-embed" in dep_names, "Lens must wait for embed instance"


def test_atlas_proxy_env_points_at_vllm_services(compose):
    env = compose["services"]["atlas-proxy"]["environment"]
    env_str = "\n".join(env) if isinstance(env, list) else str(env)
    assert "vllm-gen:8000" in env_str, "atlas-proxy LLAMA_GEN_URL must use docker DNS name"
    assert "vllm-embed:8001" in env_str, "atlas-proxy LLAMA_EMBED_URL must use docker DNS name"


def test_v3_service_env_points_at_vllm_services(compose):
    env = compose["services"]["v3-service"]["environment"]
    env_str = "\n".join(env) if isinstance(env, list) else str(env)
    assert "vllm-gen:8000" in env_str
    assert "vllm-embed:8001" in env_str


def test_hf_token_threaded_to_vllm_instances(compose):
    """vLLM downloads weights from HF on first run; gated repos need a token."""
    for svc in ("vllm-gen", "vllm-embed"):
        env = compose["services"][svc]["environment"]
        env_str = "\n".join(env) if isinstance(env, list) else str(env)
        assert "HF_TOKEN" in env_str, f"{svc} missing HF_TOKEN passthrough"


def test_no_legacy_llama_server_service(compose):
    """The llama-server service was removed in stage 6 — verify nothing
    accidentally re-introduced it."""
    services = compose["services"]
    assert "llama-server" not in services, "llama-server service must not exist"


def test_no_gguf_in_default_paths(compose):
    """Default model paths must point at AWQ directory, not GGUF."""
    for svc in ("vllm-gen", "vllm-embed"):
        cmd = compose["services"][svc]["command"]
        assert ".gguf" not in cmd, f"{svc} command references a .gguf file"
        assert "AWQ" in cmd, f"{svc} command should reference the AWQ model"


def test_lens_requirements_have_xgboost():
    """G(x) is loaded as an XGBClassifier at Lens startup. Without xgboost
    installed, the Lens silently degrades to C(x)-only scoring and the V3
    candidate-selection signal weakens. The standalone geometric-lens/
    Dockerfile installs from this file; the benchmarks/h200/Dockerfile
    installs xgboost separately. Keep them in sync."""
    reqs = (PROJECT_ROOT / "geometric-lens" / "requirements.txt").read_text()
    assert "xgboost" in reqs, "xgboost must be in geometric-lens/requirements.txt for G(x)"
    assert "scikit-learn" in reqs, "scikit-learn is needed by xgboost's sklearn-compatible API"


def test_h200_dockerfile_installs_xgboost():
    """The cloud-pod image must install xgboost too — same reason."""
    df = (PROJECT_ROOT / "benchmarks" / "h200" / "Dockerfile").read_text()
    assert "xgboost" in df, "benchmarks/h200/Dockerfile must install xgboost"
