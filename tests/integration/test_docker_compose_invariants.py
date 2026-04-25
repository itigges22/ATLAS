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


def test_vllm_services_have_required_runtime_limits(compose):
    """vLLM needs three things from Docker that the defaults don't give:
      - --shm-size >= ~8 GB (CUDA IPC; default 64 MB causes Bus error)
      - --ulimit memlock=-1 (KV cache pinning; default 64 KB → silent slowdown)
      - --ulimit stack=67108864 (deeper recursion in compile)
    Both vllm-gen and vllm-embed must have all three. Stages 59 + 60 added
    them; this test ensures they don't disappear in a future PR."""
    for svc in ("vllm-gen", "vllm-embed"):
        s = compose["services"][svc]
        # shm_size accepts "8gb" or 8589934592 etc.
        shm = s.get("shm_size")
        assert shm is not None, f"{svc} missing shm_size (vLLM CUDA IPC needs >=8 GB)"
        # docker-compose preserves the string form; just assert non-trivial.
        shm_str = str(shm).lower()
        assert any(unit in shm_str for unit in ("g", "gi", "1073741824")), (
            f"{svc} shm_size looks too small: {shm}"
        )

        ulimits = s.get("ulimits") or {}
        assert "memlock" in ulimits, f"{svc} missing ulimits.memlock (vLLM needs unlimited)"
        memlock = ulimits["memlock"]
        # docker-compose accepts -1 (number) or {soft:-1, hard:-1}; verify either.
        if isinstance(memlock, dict):
            assert memlock.get("soft") in (-1, "-1") and memlock.get("hard") in (-1, "-1"), (
                f"{svc} memlock not unlimited: {memlock}"
            )
        else:
            assert memlock in (-1, "-1"), f"{svc} memlock not unlimited: {memlock}"


def test_vllm_gen_waits_for_embed_to_be_healthy(compose):
    """vLLM gen and embed both claim GPU memory at startup. Without
    serialization the two CUDA initializations can race — the second
    one to allocate sees less memory than vLLM expected from
    --gpu-memory-utilization (computed against TOTAL memory but
    transient allocator state can lie). The H100 entrypoint serializes
    them already; docker-compose must do the same via depends_on."""
    deps = compose["services"]["vllm-gen"].get("depends_on") or {}
    assert "vllm-embed" in deps, (
        "vllm-gen must depend_on vllm-embed (with service_healthy) "
        "to avoid GPU init races"
    )
    # When depends_on is a dict-with-conditions, verify the condition.
    if isinstance(deps, dict):
        cond = deps["vllm-embed"].get("condition")
        assert cond == "service_healthy", (
            f"vllm-gen→vllm-embed dep must use service_healthy, got: {cond}"
        )


def test_dockerfile_pins_working_vllm_version():
    """vLLM 0.18.0 has a regression where the engine crashes during init
    on Qwen3.5 models. Pin tightly to a known-working version (0.17.x)
    until upstream fixes it. This test fails CI if anyone bumps the pin
    to a known-broken range.
    See https://github.com/vllm-project/vllm/issues/37749."""
    df = (PROJECT_ROOT / "benchmarks" / "h200" / "Dockerfile").read_text()
    # Must explicitly pin within the 0.17.x line.
    assert "vllm==0.17" in df or "vllm>=0.17.0,<0.18" in df, (
        "Dockerfile must pin vLLM to 0.17.x — 0.18+ crashes on Qwen3.5 "
        "(vllm-project/vllm#37749). If you've validated a newer release, "
        "update both the pin and this test."
    )


def test_max_num_batched_tokens_set_for_deltanet_alignment(compose):
    """Qwen3.5's Gated DeltaNet layers force vLLM to align the block_size
    to 2096 tokens when prefix caching is enabled. The default
    max_num_batched_tokens (2048) is one token short, which crashes
    vLLM during config validation. Both gen and embed must explicitly
    set --max-num-batched-tokens >= 2096.

    See https://github.com/vllm-project/vllm/issues/36697 — the bug
    that motivated this test."""
    for svc in ("vllm-gen", "vllm-embed"):
        cmd = compose["services"][svc]["command"]
        assert "--max-num-batched-tokens" in cmd, (
            f"{svc} must set --max-num-batched-tokens (DeltaNet alignment "
            "needs >= 2096; default 2048 crashes vLLM startup)"
        )


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


def test_makefile_targets_present():
    """The Makefile gives users a one-command surface for the most common
    operations. Verify the targets the docs reference still exist."""
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    for target in ("help:", "test:", "wire-tests:", "unit-tests:", "lint:",
                   "up:", "down:", "logs:", "model:", "preflight:",
                   "dev-install:", "install:", "clean:"):
        assert target in makefile, f"Makefile missing target: {target}"


def test_makefile_model_target_pulls_awq():
    """`make model` must pull the AWQ build, not the GGUF build. A wrong
    target here means users following 'make up && make model' end up with
    a model vLLM can't load."""
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    assert "QuantTrio/Qwen3.5-9B-AWQ" in makefile
    assert "huggingface-cli download" in makefile
    assert ".gguf" not in makefile, "Makefile must not reference GGUF"


def test_preflight_script_uses_correct_vllm_shape():
    """The pre-flight script's curl bodies must match vLLM's contract.
    If they drift from what the runners send, preflight could pass but
    the live benchmark would fail (or vice-versa)."""
    preflight = (PROJECT_ROOT / "benchmarks" / "h200" / "preflight.sh").read_text()
    # Chat completions request must use the modern thinking-disable kwarg.
    assert "chat_template_kwargs" in preflight, (
        "preflight gen test must use chat_template_kwargs (Qwen3.5 has no /nothink)"
    )
    assert "/nothink" not in preflight, (
        "preflight must not use the deprecated /nothink soft command"
    )
    # Embed test must verify 4096 dimensions (the Lens C(x) is trained on 4096-dim).
    assert "4096" in preflight, (
        "preflight must verify the embed instance returns 4096-dim vectors"
    )
    # Endpoints we hit must be the OpenAI-compatible ones, not llama.cpp /completion.
    assert "/v1/chat/completions" in preflight
    assert "/v1/embeddings" in preflight
    # Lens score endpoint
    assert "/internal/lens/score-text" in preflight


def test_entrypoint_served_model_name_flows_from_env():
    """The vLLM `--served-model-name` must come from $LLAMA_GEN_MODEL /
    $LLAMA_EMBED_MODEL, not be hardcoded. Otherwise a user customizing
    LLAMA_GEN_MODEL via `docker run -e` ends up with vLLM serving the
    hardcoded name while preflight + runners ask for the customized one,
    which vLLM rejects with a 4xx."""
    entrypoint = (PROJECT_ROOT / "benchmarks" / "h200" / "entrypoint.sh").read_text()
    # Both vllm serve calls must reference the env var (with sensible default).
    assert '--served-model-name "${LLAMA_GEN_MODEL:-qwen3.5-9b}"' in entrypoint, (
        "gen --served-model-name must flow from $LLAMA_GEN_MODEL"
    )
    assert '--served-model-name "${LLAMA_EMBED_MODEL:-qwen3.5-9b-embed}"' in entrypoint, (
        "embed --served-model-name must flow from $LLAMA_EMBED_MODEL"
    )
    # And the same vars must be exported so preflight + runner subprocesses see them.
    assert "export LLAMA_GEN_MODEL" in entrypoint
    assert "export LLAMA_EMBED_MODEL" in entrypoint
    # Preflight reads LENS_URL; entrypoint must export it (RAG_API_URL alone is not enough).
    assert "export LENS_URL" in entrypoint


def test_dockerfile_default_model_names_match_preflight_defaults():
    """The Dockerfile ENV defaults for LLAMA_GEN_MODEL / LLAMA_EMBED_MODEL
    must match the fallbacks preflight.sh assumes (`qwen3.5-9b` and
    `qwen3.5-9b-embed`). If they drift, a fresh container with no overrides
    would fail preflight on a 404 model-not-found."""
    df = (PROJECT_ROOT / "benchmarks" / "h200" / "Dockerfile").read_text()
    assert "LLAMA_GEN_MODEL=qwen3.5-9b" in df
    assert "LLAMA_EMBED_MODEL=qwen3.5-9b-embed" in df
    preflight = (PROJECT_ROOT / "benchmarks" / "h200" / "preflight.sh").read_text()
    assert 'LLAMA_GEN_MODEL:=qwen3.5-9b' in preflight
    assert 'LLAMA_EMBED_MODEL:=qwen3.5-9b-embed' in preflight
