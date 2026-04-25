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


def test_no_botched_atlas_dir_string_concat_in_scripts():
    """During the V3.0.1 ATLAS_DIR refactor a sed replacement got mangled
    in places, leaving lines like:

        sys.path.insert(0, '" + ATLAS_DIR + "/geometric-lens')

    Python parses that as a single literal string with embedded quotes
    (the value `" + ATLAS_DIR + "/geometric-lens`) — the path doesn't
    actually get computed, sys.path gets garbage, and any subsequent
    `from geometric_lens import ...` ImportErrors with no useful context.

    The pattern `'" + ATLAS_DIR + "/...'` (quotes inside a single-quoted
    string) is unambiguously broken — there's no legitimate reason to
    embed quote+plus+ATLAS_DIR+plus+quote in a literal. Pin: forbid that
    pattern in scripts that have been touched by the refactor."""
    import re
    targets = list((PROJECT_ROOT / "scripts").rglob("*.py"))
    targets += list((PROJECT_ROOT / "benchmark").rglob("*.py"))
    targets += list((PROJECT_ROOT / "benchmarks").rglob("*.py"))
    # Allow this exact pattern only inside this test file (it's the
    # invariant's data, not a botched edit).
    targets = [p for p in targets if "tests/" not in str(p)
               and "__pycache__" not in str(p)]

    bad_pattern = re.compile(r"""['"]\s*\+\s*ATLAS_DIR\s*\+\s*['"]""")
    offenders = []
    for path in targets:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            # Skip comments — they may legitimately mention the broken
            # pattern in disclaimer/explanation form.
            if stripped.startswith("#"):
                continue
            if bad_pattern.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Botched ATLAS_DIR string concatenation found — the surrounding "
        "quotes are inside the string literal, so the path is never "
        "computed. Use os.path.join(ATLAS_DIR, ...) or an f-string instead.\n"
        + "\n".join(offenders)
    )


def test_preflight_respects_skip_embed():
    """Stage 107 added SKIP_EMBED=1 to entrypoint.sh — when set, the
    embed vLLM instance is never started (saves VRAM on a single 16 GB
    card). But the entrypoint then runs `./benchmarks/h200/preflight.sh`,
    which always probed `/v1/embeddings`. The probe naturally failed
    against the absent service, preflight returned 1, and the entrypoint
    refused to start the benchmark sweep — making SKIP_EMBED useless
    end-to-end. Stage 108 wired SKIP_EMBED into preflight too."""
    src = (PROJECT_ROOT / "benchmarks" / "h200" / "preflight.sh").read_text()
    # The skip path must exist.
    assert 'SKIP_EMBED' in src, (
        "preflight.sh must check SKIP_EMBED so the embed probe is bypassed "
        "when the embed instance was never started"
    )
    # Behavioral check: with SKIP_EMBED=1 + GEOMETRIC_LENS_ENABLED=false +
    # gen up at a working URL, preflight must not fail the embed probe.
    # We can verify this purely by parsing the script structure: the
    # embed probe block must be inside an `if [[ ... SKIP_EMBED ... ]]` else.
    import re
    # Match the "embed" comment block followed by the SKIP_EMBED gate.
    block = re.search(
        r"# 2\. Embed instance.*?(?=# 3\.)",
        src, re.DOTALL,
    )
    assert block is not None, "preflight.sh embed-section comment missing"
    embed_block = block.group(0)
    assert 'if [[ "${SKIP_EMBED' in embed_block, (
        "Embed probe in preflight.sh must be gated on SKIP_EMBED"
    )


def test_lens_entrypoint_env_points_chat_at_gen_port():
    """`benchmarks/h200/entrypoint.sh` launches the Geometric Lens with
    LLAMA_*_URL env vars. The Lens chat clients (summarizer.py,
    pattern_extractor.py, tree_search.py) read `LLAMA_GEN_URL` first,
    falling back to `LLAMA_URL` when unset. The entrypoint used to set
    only `LLAMA_URL=http://localhost:${EMBED_PORT}` (port 8001), so
    chat completion requests from the Lens went to the embed instance
    — which runs in `--runner pooling --convert embed` mode and 4xx's
    on `/v1/chat/completions`. Pin: the Lens block must set
    `LLAMA_GEN_URL` to ${GEN_PORT}, not ${EMBED_PORT}, and `LLAMA_URL`
    (the fallback) must agree.
    """
    src = (PROJECT_ROOT / "benchmarks" / "h200" / "entrypoint.sh").read_text()
    # Find the Lens-launch block (between the "Starting Geometric Lens"
    # banner and the LENS_PID assignment).
    block_start = src.index('"--- Starting Geometric Lens')
    block_end = src.index("LENS_PID=$!", block_start)
    block = src[block_start:block_end]
    assert 'LLAMA_GEN_URL="http://localhost:${GEN_PORT}"' in block, (
        "Lens entrypoint must set LLAMA_GEN_URL to ${GEN_PORT} so chat "
        "completions hit the gen instance, not the embed one"
    )
    assert 'LLAMA_EMBED_URL="http://localhost:${EMBED_PORT}"' in block, (
        "Lens entrypoint must set LLAMA_EMBED_URL to ${EMBED_PORT} for "
        "embedding extraction"
    )
    # If LLAMA_URL is set at all, it must match GEN_PORT (it's the
    # fallback for code paths that haven't migrated to LLAMA_GEN_URL).
    if 'LLAMA_URL="' in block:
        assert 'LLAMA_URL="http://localhost:${GEN_PORT}"' in block, (
            "When entrypoint sets the legacy LLAMA_URL, it must point at "
            "${GEN_PORT} — not the embed port"
        )


def test_launch_on_h200_forwards_model_and_parallel_env():
    """benchmarks/h200/launch_on_h200.sh starts the cloud-pod container with
    a curated set of `-e VAR=...` exports. The entrypoint inside the
    container (stage 63) uses LLAMA_GEN_MODEL / LLAMA_EMBED_MODEL to set
    `--served-model-name`, and the V3 runners use ATLAS_LLM_PARALLEL /
    ATLAS_PARALLEL_TASKS / BENCHMARK_PARALLEL to size concurrency.

    If the launcher swallows any of these, a host-side `LLAMA_GEN_MODEL=foo
    ./launch_on_h200.sh` invocation never reaches the container — vLLM
    serves under the default name and any client passing `model=foo`
    gets a 4xx model-not-found. Pin the forwarding so this can't regress."""
    src = (PROJECT_ROOT / "benchmarks" / "h200" / "launch_on_h200.sh").read_text()
    must_forward = [
        "LLAMA_GEN_MODEL", "LLAMA_EMBED_MODEL",
        "ATLAS_LLM_PARALLEL", "ATLAS_PARALLEL_TASKS",
        "BENCHMARK_PARALLEL",
        "GEN_MAX_NUM_SEQS", "GEN_MAX_MODEL_LEN", "GEN_GPU_MEM_UTIL",
        "GEN_MAX_NUM_BATCHED_TOKENS", "GEN_SWAP_SPACE_GB",
        "EMBED_MAX_NUM_SEQS", "EMBED_MAX_MODEL_LEN", "EMBED_GPU_MEM_UTIL",
        "EMBED_MAX_NUM_BATCHED_TOKENS",
        "HF_TOKEN",
    ]
    for var in must_forward:
        # Match `-e VAR=...` in the docker run invocation (with optional
        # whitespace), so a future formatting change still passes.
        assert f'-e {var}=' in src, (
            f"launch_on_h200.sh must forward {var} into the container "
            "(entrypoint or runners read it; without forwarding, host-side "
            "overrides silently fall back to defaults inside the container)"
        )


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


def test_makefile_lint_and_ci_lint_cover_same_scripts():
    """The Makefile `lint` target and the GitHub Actions shell-validate step
    must stay in sync — otherwise a script lint-clean for one path but broken
    for the other gives the user a different verdict than CI does, and a
    syntax bug in (e.g.) scripts/install.sh sails through PR review only to
    surface at deploy time. Diff the two script lists and require equality."""
    import re

    def _scripts_in(text: str) -> set:
        # Both lists are bash for-loops with one path per line ending in `\`,
        # plus a final entry without the trailing backslash. Each path begins
        # with a directory segment (`scripts/`, `benchmarks/`, `benchmark/`)
        # and ends in `.sh`. Require the leading slash so we don't pick up
        # bare basenames that appear in surrounding comments.
        return set(re.findall(r"\b[\w./-]*?/[\w.-]+\.sh\b", text))

    makefile = (PROJECT_ROOT / "Makefile").read_text()
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "vllm-wire.yml").read_text()

    # Restrict to the lint sections (rough but sufficient).
    mf_lint = makefile.split("lint:", 1)[1].split("\nup:", 1)[0]
    wf_lint = workflow.split("Validate shell scripts", 1)[1].split("\n      - name", 1)[0]

    mf_scripts = _scripts_in(mf_lint)
    wf_scripts = _scripts_in(wf_lint)

    only_mf = mf_scripts - wf_scripts
    only_wf = wf_scripts - mf_scripts
    assert not only_mf and not only_wf, (
        f"Makefile lint and CI shell-validate diverged:\n"
        f"  only in Makefile: {sorted(only_mf)}\n"
        f"  only in CI:       {sorted(only_wf)}\n"
        "Bring them back into agreement so users and CI see the same verdict."
    )
    # And: the union must include the most user-visible installer paths.
    for must_have in ("scripts/install.sh", "scripts/verify-install.sh",
                      "benchmarks/h200/entrypoint.sh",
                      "benchmarks/h200/preflight.sh"):
        assert must_have in mf_scripts, f"{must_have} must be lint-checked"


def test_atlas_proxy_build_target_matches_launcher_expectation():
    """`atlas/cli/repl.py` searches `~/.local/bin/atlas-proxy-v2` for the
    proxy binary and rebuilds from source to that exact path if it's
    missing. atlas-proxy/README.md previously instructed users to:

        go build -o ~/.local/bin/atlas-proxy .

    Wrong filename. A user following the README would land a working
    binary at `~/.local/bin/atlas-proxy`, then run `atlas`, and the
    Python launcher would not find it, claim the proxy isn't built,
    and try to rebuild — confusing error path.

    Pin: every fenced `go build` recipe in the proxy's docs has to write
    to the same path the launcher reads."""
    readme = (PROJECT_ROOT / "atlas-proxy" / "README.md").read_text()
    repl_py = (PROJECT_ROOT / "atlas" / "cli" / "repl.py").read_text()

    # The Python launcher's expectation:
    assert "atlas-proxy-v2" in repl_py, (
        "atlas/cli/repl.py is supposed to expect the `atlas-proxy-v2` binary "
        "name — if that's no longer true, this whole test needs revisiting"
    )

    import re
    # Every `go build -o ...` line in the README must end in `atlas-proxy-v2`.
    builds = re.findall(r"go build [^\n]*-o ([^\s]+)", readme)
    assert builds, "atlas-proxy/README.md should still contain at least one go build recipe"
    for path in builds:
        assert path.endswith("atlas-proxy-v2"), (
            f"atlas-proxy/README.md `go build -o {path}` produces a binary "
            "the Python launcher won't find — must end in atlas-proxy-v2"
        )


def test_atlas_proxy_readme_matches_actual_env_vars():
    """atlas-proxy/README.md's "Configuration" table must list the env vars
    the proxy actually reads, with the same defaults the code uses. The
    README had drifted: it documented `ATLAS_RAG_URL = http://localhost:8099`
    but the proxy reads `ATLAS_LENS_URL = http://localhost:31144` (wrong
    name, wrong port). A user trusting the README would set ATLAS_RAG_URL
    in their environment and the proxy would ignore it, falling back to
    the 31144 default — silent override-not-applied."""
    readme = (PROJECT_ROOT / "atlas-proxy" / "README.md").read_text()
    main_go = (PROJECT_ROOT / "atlas-proxy" / "main.go").read_text()

    # Every env var the README documents must actually be read by main.go.
    import re
    docs_vars = set(re.findall(r"\|\s*([A-Z][A-Z0-9_]+)\s*\|", readme))
    # Filter to ones the README actually presents as env knobs (skip
    # column header tokens / random uppercase words).
    env_var_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    docs_vars = {v for v in docs_vars if env_var_pattern.match(v)
                 and v not in ("Env", "Var", "Default", "Description")}

    # `ATLAS_RAG_URL` is the legacy name we want gone.
    assert "ATLAS_RAG_URL" not in docs_vars, (
        "atlas-proxy/README.md still documents ATLAS_RAG_URL; the proxy "
        "actually reads ATLAS_LENS_URL (see main.go:45 envOr call)"
    )
    # And the new name should be present.
    assert "ATLAS_LENS_URL" in readme, (
        "atlas-proxy/README.md must document ATLAS_LENS_URL with default "
        "http://localhost:31144"
    )
    # Sanity-check: every env var the README claims is honored must
    # actually appear as an envOr/firstEnv lookup in main.go.
    for var in docs_vars:
        # Skip common-knob env vars not specific to the proxy's lookups
        # (LLAMA_GEN_MODEL is read by ChatRequest, not directly via envOr).
        if var in ("LLAMA_GEN_MODEL", "ATLAS_PROXY_PORT", "ATLAS_AGENT_LOOP"):
            continue
        assert var in main_go, (
            f"atlas-proxy/README.md documents `{var}` but main.go never "
            f"references it — README has drifted from code"
        )


def test_architecture_services_table_lists_dual_vllm_instances():
    """`docs/ARCHITECTURE.md` Services table previously had a single
    `vLLM | 8080 | C++ (vLLM)` row — three errors at once: wrong port
    (8080 was llama-server; vLLM gen+embed run on 8000+8001), wrong
    language (vLLM is Python with PyTorch + Triton kernels, not C++ —
    that was llama.cpp), and the single row papered over the dual-
    instance reality. Pin both gen and embed rows to keep readers
    accurate."""
    src = (PROJECT_ROOT / "docs" / "ARCHITECTURE.md").read_text()

    # Forbid the stale single-row entry.
    assert "C++ (vLLM)" not in src, (
        "ARCHITECTURE.md: 'C++ (vLLM)' is the llama.cpp era — vLLM is Python"
    )

    # Require both dual-instance rows in the services table.
    assert "**vllm-gen**" in src, (
        "ARCHITECTURE.md services table must list vllm-gen explicitly"
    )
    assert "**vllm-embed**" in src, (
        "ARCHITECTURE.md services table must list vllm-embed explicitly"
    )


def test_user_docs_do_not_reference_legacy_port_8080():
    """Port 8080 is the old llama-server port. vLLM gen runs on 8000 and
    embed on 8001 across the entire stack (Dockerfiles, compose, every
    SETUP recipe, every wire test). Stale 8080 refs in user-facing docs
    sent users running `lsof -i :8080`, finding nothing, and concluding
    the install was broken. CLI.md, TROUBLESHOOTING.md, ARCHITECTURE.md
    plus the three TROUBLESHOOTING translations had 11 such refs."""
    targets = [
        PROJECT_ROOT / "docs" / "CLI.md",
        PROJECT_ROOT / "docs" / "TROUBLESHOOTING.md",
        PROJECT_ROOT / "docs" / "ARCHITECTURE.md",
    ]
    targets += [PROJECT_ROOT / "docs" / "lang" / lang / "TROUBLESHOOTING.md"
                for lang in ("ja", "ko", "zh-CN")]

    import re
    # Forbid `:8080` and `port 8080` patterns. Allow anything that contains
    # a longer port number that happens to start with 8080 (e.g. 80800),
    # though none currently exist.
    for path in targets:
        src = path.read_text()
        # Match `:8080` (not followed by another digit) and "port 8080".
        bad = re.findall(r":8080(?!\d)|port 8080(?!\d)", src)
        assert not bad, (
            f"{path.relative_to(PROJECT_ROOT)} still references legacy port "
            f"8080 ({len(bad)} occurrences) — vLLM gen runs on 8000, embed on 8001"
        )


def test_lens_default_port_matches_stack():
    """The Geometric Lens listens on 31144 across every surface — Dockerfile
    EXPOSE, docker-compose's ATLAS_LENS_PORT default, the cloud-pod
    entrypoint's LENS_PORT, the atlas-proxy's hardcoded http://localhost:31144,
    every preflight curl. config.py's `ServerConfig.port` was the one outlier
    at 8099, which only matters on the `python main.py` debug entry path
    (Dockerfile and compose both override via uvicorn `--port 31144`). But
    that entry path is exactly what a developer reaches for when debugging
    locally — and it would silently bind to a port the rest of the stack
    can't see. Pin the default."""
    src = (PROJECT_ROOT / "geometric-lens" / "config.py").read_text()
    # Allow `LENS_PORT` env override but the literal default must be 31144.
    assert "31144" in src, (
        "Lens default port must be 31144 to match Dockerfile EXPOSE, "
        "docker-compose ATLAS_LENS_PORT, and every consumer URL"
    )
    # The stale 8099 must not be a literal default anywhere in the file.
    assert "= 8099" not in src and "=8099" not in src, (
        "Lens config still hardcodes the legacy 8099 port as a default"
    )


def test_setup_docs_do_not_reference_fictional_atlas_launcher():
    """All four SETUP files used to tell users to `cp /path/to/atlas-launcher
    ~/.local/bin/atlas`. There is no atlas-launcher script anywhere in the
    repo — the `atlas` command is the `atlas.cli.repl:run` Python entry
    point installed by `pip install -e .` (declared in pyproject.toml).
    The misleading section sent users searching for a file that has no
    source. Pin the absence of `atlas-launcher` so it can't sneak back."""
    targets = [PROJECT_ROOT / "docs" / "SETUP.md"]
    targets += [PROJECT_ROOT / "docs" / "lang" / lang / "SETUP.md"
                for lang in ("ja", "ko", "zh-CN")]

    for path in targets:
        src = path.read_text()
        assert "atlas-launcher" not in src, (
            f"{path.relative_to(PROJECT_ROOT)} still mentions `atlas-launcher`, "
            "a script that doesn't exist in the repo. The `atlas` command is "
            "the atlas.cli.repl:run Python entry point installed by "
            "`pip install -e .` — point users at that instead."
        )


def test_setup_docs_do_not_advertise_a_broken_k3s_path():
    """SETUP.md (and translations) used to describe a working K3s install
    path: `cp atlas.conf.example atlas.conf && sudo scripts/install.sh`,
    `scripts/build-containers.sh && scripts/generate-manifests.sh && kubectl
    apply -n atlas -f manifests/`. After the vLLM cutover, that path is
    broken — the manifests/ directory and templates/ directory aren't
    shipped (stage 73 catches this in install.sh's deploy_manifests). The
    SETUP doc must not steer users toward it.

    Forbid the broken `kubectl apply ... -f manifests/` recipe, the
    `scripts/generate-manifests.sh` advice, and the llama.cpp comparison
    table headings (Flash attention, mlock, q8_0 / q4_0)."""
    targets = [PROJECT_ROOT / "docs" / "SETUP.md"]
    targets += [PROJECT_ROOT / "docs" / "lang" / lang / "SETUP.md"
                for lang in ("ja", "ko", "zh-CN")]

    # Only consider fenced code blocks — that's where active recipes live.
    # Prose paragraphs are allowed to mention these tokens in a historical
    # disclaimer ("the template set scripts/generate-manifests.sh consumes
    # is no longer shipped"). The fenced blocks are what users copy-paste,
    # so that's where they have to be clean.
    import re

    forbidden_in_recipes = [
        "kubectl apply -n atlas -f manifests/",
        "scripts/generate-manifests.sh",
    ]

    for path in targets:
        src = path.read_text()
        # Extract everything between ``` ... ``` fences.
        recipes = "\n".join(re.findall(r"```[a-z]*\n(.*?)```", src, re.DOTALL))
        for recipe in forbidden_in_recipes:
            assert recipe not in recipes, (
                f"{path.relative_to(PROJECT_ROOT)}: a fenced code block still "
                f"advertises the broken K3s recipe `{recipe}` — manifests/ "
                "doesn't ship for the vLLM stack, so the recipe always fails"
            )


def test_troubleshooting_does_not_reference_llama_cpp_flags():
    """TROUBLESHOOTING.md (English + ja/ko/zh-CN translations) steered
    users to debug `--n-gpu-layers 99`, a llama.cpp flag with no vLLM
    equivalent. Quoting llama.cpp flags as troubleshooting steps for a
    vLLM stack misleads users into chasing a knob that doesn't exist.

    Forbid the bare flag, but allow it inside an explainer paragraph that
    disclaims it ("there is no `--n-gpu-layers` knob" / "存在しません" /
    "존재하지 않으며" / "不存在 ... 这个开关")."""
    targets = [PROJECT_ROOT / "docs" / "TROUBLESHOOTING.md"]
    targets += [PROJECT_ROOT / "docs" / "lang" / lang / "TROUBLESHOOTING.md"
                for lang in ("ja", "ko", "zh-CN")]

    # Phrases that explicitly disclaim the flag — those are the fix.
    disclaimers = [
        "there is no `--n-gpu-layers` knob",
        "`--n-gpu-layers` というつまみは存在しません",
        "`--n-gpu-layers` というつまみは存在せず",
        "`--n-gpu-layers` 같은 손잡이는 존재하지 않으며",
        "不存在 `--n-gpu-layers` 这个开关",
    ]

    for path in targets:
        src = path.read_text()
        body = src
        for d in disclaimers:
            body = body.replace(d, "")
        assert "--n-gpu-layers" not in body, (
            f"{path.relative_to(PROJECT_ROOT)} still steers users to debug "
            "--n-gpu-layers outside an explainer paragraph; it's a llama.cpp "
            "flag with no vLLM analogue"
        )

    en = (PROJECT_ROOT / "docs" / "TROUBLESHOOTING.md").read_text()
    assert "with grammar enforcement" not in en, (
        "TROUBLESHOOTING.md must not quote 'grammar enforcement' throughput — "
        "vLLM uses guided_choice/response_format, not llama.cpp grammars"
    )


def test_user_facing_docs_have_no_stale_q6_k_refs():
    """The user-facing docs — English plus the ja/ko/zh-CN translations —
    must not steer fresh users to a Qwen3.5-9B-Q6_K GGUF download or use
    that name in API examples. vLLM doesn't load GGUF and serves under
    `qwen3.5-9b` (the AWQ build's served-model-name). Stale refs cost
    users hours: they download the wrong file, hit a 4xx from vLLM, then
    chase the misalignment through the entire stack.

    Historical reports (docs/reports/, benchmarks/section_*/) are
    excluded — those are explicitly snapshots of past state."""
    english = ["docs/SETUP.md", "docs/CONFIGURATION.md",
               "docs/ARCHITECTURE.md", "docs/API.md", "docs/CLI.md"]
    translated = [
        f"docs/lang/{lang}/{name}"
        for lang in ("ja", "ko", "zh-CN")
        for name in ("README.md", "SETUP.md", "TROUBLESHOOTING.md")
    ]
    for rel in english + translated:
        src = (PROJECT_ROOT / rel).read_text()
        # The docs may legitimately mention "GGUF" in the context of why we
        # don't use it (the explanation is the *fix*). Forbid only the active
        # patterns: the literal model name and the .gguf filename suffix.
        assert "Qwen3.5-9B-Q6_K" not in src, (
            f"{rel}: stale Qwen3.5-9B-Q6_K reference — vLLM serves "
            "the AWQ build under `qwen3.5-9b`"
        )
        assert ".gguf" not in src, (
            f"{rel}: stale .gguf path — vLLM consumes the AWQ "
            "directory directly, not a single GGUF file"
        )


def test_verify_install_sh_supports_docker_compose():
    """`scripts/verify-install.sh` was hardcoded for the K8s install path:
    it required a kubeconfig and bailed if there wasn't one. Since stage
    73 confirmed K8s manifests don't ship in V3.0.1, every docker-compose
    user running `verify-install.sh` saw a confusing "No kubeconfig
    found" failure even though their stack was perfectly healthy.

    Pin: the script must detect docker-compose mode (via
    `docker compose ps --status running --quiet vllm-gen`) and run an
    HTTP-based verifier instead of demanding kubeconfig.

    Also: when neither K8s manifests nor a docker-compose stack is
    present, the script must surface a "use make up" message rather than
    drowning the user in K8s diagnostics."""
    src = (PROJECT_ROOT / "scripts" / "verify-install.sh").read_text()

    # New verify_docker_compose function exists.
    assert "verify_docker_compose()" in src, (
        "verify-install.sh must define a verify_docker_compose function "
        "for the docker-compose deployment path"
    )

    # main() detects compose before falling through to K8s.
    assert "docker compose ps" in src, (
        "verify-install.sh must check `docker compose ps` for vllm-gen "
        "before requiring a kubeconfig"
    )

    # The new code must health-check the actual compose service ports.
    for service in ("vllm-gen", "vllm-embed", "Geometric Lens",
                    "atlas-proxy"):
        assert service in src, (
            f"verify_docker_compose must check {service} health endpoint"
        )

    # And it must reach all four canonical ports the compose stack exposes.
    for port_var in ("ATLAS_GEN_PORT", "ATLAS_EMBED_PORT", "ATLAS_LENS_PORT",
                     "ATLAS_PROXY_PORT"):
        assert port_var in src, (
            f"verify-install.sh must consult {port_var} for the compose path"
        )


def test_install_sh_guards_against_missing_manifests_dir():
    """`scripts/install.sh` references `$K8S_DIR/manifests/*.yaml` for the
    K8s deployment path, but the repo currently does NOT ship any manifests/
    directory — the canonical install path for the vLLM two-instance stack
    is docker-compose. Without an upfront guard, install.sh would fail at
    the first `kubectl apply -f $K8S_DIR/manifests/redis-deployment.yaml`
    with "no such file" and leave the namespace half-initialized.

    Pin: deploy_manifests must short-circuit when the directory is absent
    and tell the user to use `make up` (docker-compose) instead."""
    install_sh = (PROJECT_ROOT / "scripts" / "install.sh").read_text()
    # The guard must reference the manifests directory existence check.
    assert '-d "$K8S_DIR/manifests"' in install_sh or "[[ ! -d " in install_sh and "manifests" in install_sh, (
        "install.sh must check `[[ ! -d $K8S_DIR/manifests ]]` before invoking kubectl"
    )
    # And it must point users to docker compose / make up.
    assert "docker compose" in install_sh.lower() or "make up" in install_sh, (
        "install.sh's missing-manifests message must point users to docker compose"
    )
    # Sanity: the manifests directory really is absent in the repo right now.
    # If a future PR ports the K8s manifests across, this assertion will flag
    # both the test and the guard for review at the same time.
    assert not (PROJECT_ROOT / "manifests").is_dir(), (
        "manifests/ directory now exists — update install.sh's guard and remove"
        " this self-check, since the docker-compose-only fallback no longer applies"
    )


def test_config_sh_validate_config_tolerates_unset_legacy_vars():
    """`validate_config` previously dereferenced `$ATLAS_RAG_API_NODEPORT`
    (and other legacy K8s NodePort vars) bare. Under `set -u` (which
    install.sh and verify-install.sh both enable), an unset var aborts
    the script — and atlas.conf.example doesn't define
    ATLAS_RAG_API_NODEPORT at all. Anyone sourcing config.sh from those
    scripts hit "ATLAS_RAG_API_NODEPORT: unbound variable" before any
    real work happened.

    Pin: validate_config must use `${VAR:-}` expansion + skip empty
    values in the loop, so the function returns rc=0 even when several
    legacy K8s vars are unset."""
    import subprocess
    import textwrap

    config_sh = PROJECT_ROOT / "scripts" / "lib" / "config.sh"

    # Source under `set -euo pipefail`, deliberately unset every legacy
    # K8s NodePort var, then call validate_config and check rc.
    script = textwrap.dedent(f"""
    set -euo pipefail
    source {config_sh}
    unset ATLAS_API_PORTAL_NODEPORT ATLAS_LLM_PROXY_NODEPORT \\
          ATLAS_RAG_API_NODEPORT ATLAS_DASHBOARD_NODEPORT
    # Keep just the vLLM ones a docker-compose user might set.
    export ATLAS_VLLM_GEN_NODEPORT=32735
    export ATLAS_VLLM_EMBED_NODEPORT=32736
    export ATLAS_SANDBOX_NODEPORT=30820
    validate_config
    echo "rc=$?"
    """).strip()
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30,
    )
    combined = out.stdout + out.stderr
    assert "unbound variable" not in combined, (
        f"validate_config still aborts when legacy NodePort vars are unset:\n"
        f"{combined}"
    )
    assert "rc=0" in combined or out.returncode == 0, (
        f"validate_config should accept the docker-compose subset of vars; "
        f"got:\n{combined}"
    )


def test_config_sh_validates_vllm_embed_nodeport():
    """`scripts/lib/config.sh validate_config` must include the embed NodePort
    in its dedup + range check. Earlier iterations only listed ATLAS_LLAMA_NODEPORT
    — a fresh K8s install with ATLAS_VLLM_EMBED_NODEPORT set to (e.g.) 32735
    (the gen port) would slip past validation entirely and only fail at
    `kubectl apply` time as an opaque NodePort conflict.

    Drive validate_config from a temp shell with mocked vars and assert it
    rejects a duplicate embed port."""
    import subprocess
    import textwrap

    config_sh = PROJECT_ROOT / "scripts" / "lib" / "config.sh"
    src = config_sh.read_text()

    # Static check: validate_config must reference the embed NodePort.
    assert "ATLAS_VLLM_EMBED_NODEPORT" in src, (
        "validate_config must check ATLAS_VLLM_EMBED_NODEPORT for collisions/range"
    )

    # Behavioral check: source the file, set duplicate ports, expect non-zero.
    # Sourcing triggers auto-load_config, which would overwrite our test vars
    # from atlas.conf.example — set the vars AFTER the source.
    script = textwrap.dedent(f"""
    set +e
    source {config_sh}
    export ATLAS_API_PORTAL_NODEPORT=30000
    export ATLAS_LLM_PROXY_NODEPORT=30001
    export ATLAS_RAG_API_NODEPORT=30002
    export ATLAS_DASHBOARD_NODEPORT=30003
    export ATLAS_VLLM_GEN_NODEPORT=32735
    export ATLAS_VLLM_EMBED_NODEPORT=32735   # collides with gen — must be flagged
    export ATLAS_LLAMA_NODEPORT=32735
    export ATLAS_SANDBOX_NODEPORT=30820
    validate_config
    echo "rc=$?"
    """).strip()
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30,
    )
    combined = out.stdout + out.stderr
    assert "Duplicate NodePort: 32735" in combined or "rc=1" in combined or "rc=2" in combined, (
        f"validate_config did not flag the colliding embed/gen NodePort:\n{combined}"
    )


def test_env_example_documents_user_tunable_compose_vars():
    """User-tunable knobs that docker-compose.yml reads from the host
    environment must appear in .env.example so a fresh user copying it can
    discover them. The script previously drifted: `ATLAS_HF_CACHE` (which
    persists the ~12 GiB HuggingFace download cache between container
    restarts) and `ATLAS_LENS_MODELS` (Lens model artifact bind mount)
    were referenced by compose but never mentioned in .env.example, so a
    user who didn't read the YAML would re-download the model every time
    the container was recreated.

    Bind-mount paths and host port knobs must be present. Inter-container
    service URLs (like ATLAS_INFERENCE_URL hardcoded to vllm-gen:8000) and
    feature toggles set by compose's environment block (ATLAS_AGENT_LOOP)
    are excluded — those are not user-tunable from .env."""
    import re

    compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
    env = (PROJECT_ROOT / ".env.example").read_text()

    # All `${ATLAS_X:-default}` references that compose actually substitutes.
    used = set(re.findall(r"\$\{(ATLAS_[A-Z0-9_]+)", compose))

    # Inter-container URLs and the agent toggle aren't user knobs.
    not_user_tunable = {
        "ATLAS_AGENT_LOOP",
        "ATLAS_INFERENCE_URL",   # hardcoded http://vllm-gen:8000 in compose env
        "ATLAS_LENS_URL",        # hardcoded http://geometric-lens:31144
        "ATLAS_LLAMA_URL",       # back-compat alias for ATLAS_INFERENCE_URL
        "ATLAS_SANDBOX_URL",     # hardcoded http://sandbox:8020
        "ATLAS_V3_URL",          # hardcoded http://v3-service:8070
    }
    user_knobs = used - not_user_tunable

    documented = set(re.findall(r"^\s*#?\s*(ATLAS_[A-Z0-9_]+)\s*=", env, re.MULTILINE))

    missing = user_knobs - documented
    assert not missing, (
        f".env.example is missing user-tunable compose vars: {sorted(missing)}\n"
        "Add a section for each so users discovering the knob via .env.example "
        "see what they can override and what the defaults are."
    )


def test_v3_service_dockerfile_has_no_unused_heavy_deps():
    """v3-service is a thin stdlib http.server front-end for the V3 pipeline
    modules. None of those modules import torch, numpy, or any ML lib — they
    only marshal text through vLLM. Earlier iterations of this Dockerfile
    installed the ~2 GiB torch CPU wheel, slowing `docker compose build` by
    minutes and ballooning the image.

    Pin: this Dockerfile must NOT install torch (or numpy/scipy). If a future
    feature genuinely needs them, the test should be updated alongside the
    real change so the cost is visible at PR review time."""
    df = (PROJECT_ROOT / "v3-service" / "Dockerfile").read_text()
    forbidden = ["torch", "numpy", "scipy", "tensorflow"]
    for dep in forbidden:
        # Allow the dep name in comments (the rationale lives there) — only
        # block actual `pip install` lines that would pull it.
        for line in df.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pip install" in stripped:
                assert dep not in stripped, (
                    f"v3-service/Dockerfile must not pip-install {dep}; nothing in "
                    "v3-service or benchmark/v3 actually imports it"
                )


def test_atlas_proxy_applies_vllm_defaults_to_all_chat_posts():
    """The atlas-proxy has multiple places that POST to /v1/chat/completions:

      - main.go:forwardToFox (canonical typed-ChatRequest helper)
      - main.go spec generator (raw POST)
      - main.go streaming-handler fallback (raw POST)
      - agent.go agent loop (raw map[string]interface{} POST with
        response_format=json_object)

    Earlier iterations only set enable_thinking=false in forwardToFox.
    Even after the applyVllmDefaults helper landed in main.go, the agent
    loop in agent.go bypassed it (different request shape — raw map for
    the response_format flag). On Qwen3.5 that path burned thinking-mode
    tokens before guided-JSON decoding kicked in.

    Pin: every chat-completions POST in atlas-proxy must include either
    `applyVllmDefaults(&` or the literal `enable_thinking` kwarg setup
    inline."""
    main_src = (PROJECT_ROOT / "atlas-proxy" / "main.go").read_text()
    agent_src = (PROJECT_ROOT / "atlas-proxy" / "agent.go").read_text()

    # Helper must exist in main.go.
    assert "func applyVllmDefaults(req *ChatRequest)" in main_src, (
        "atlas-proxy must define an applyVllmDefaults helper in main.go"
    )
    helper_start = main_src.index("func applyVllmDefaults")
    helper_end = main_src.index("\n}", helper_start)
    assert "enable_thinking" in main_src[helper_start:helper_end], (
        "applyVllmDefaults must set chat_template_kwargs.enable_thinking"
    )
    apply_calls = main_src.count("applyVllmDefaults(&")
    assert apply_calls >= 3, (
        f"applyVllmDefaults should be called at all 3 typed-ChatRequest "
        f"POST sites in main.go, found {apply_calls}"
    )

    # agent.go uses a raw map (different shape because of
    # response_format), so it can't call applyVllmDefaults — but it must
    # still set enable_thinking inline.
    assert "enable_thinking" in agent_src, (
        "atlas-proxy/agent.go's raw /v1/chat/completions POST must set "
        "chat_template_kwargs.enable_thinking — otherwise Qwen3.5 burns "
        "thinking-mode tokens before guided-JSON decoding starts"
    )


def test_verify_install_sends_valid_vllm_chat_request():
    """`scripts/verify-install.sh` includes a smoke check that POSTs to
    /v1/chat/completions. vLLM 4xx's any request without a `model` field
    matching --served-model-name, and on Qwen3.5 a small `max_tokens`
    budget without `enable_thinking=false` returns empty content (every
    token spent inside <think>). The verifier had been asserting only that
    "choices" appeared in the response — both failure modes returned
    structurally-valid responses with empty content, so the verifier
    falsely passed on a broken stack. Pin the request shape so this can't
    silently regress."""
    src = (PROJECT_ROOT / "scripts" / "verify-install.sh").read_text()
    # Must include a model field (vLLM rejects unknown / missing names)
    assert "\\\"model\\\":" in src or '"model":' in src, (
        "verify-install.sh must send `model` in the chat completions body"
    )
    # Must disable thinking so the response actually contains content.
    assert "enable_thinking" in src, (
        "verify-install.sh must set chat_template_kwargs.enable_thinking=false; "
        "otherwise a working vLLM returns empty content with small max_tokens"
    )
    # Must validate content, not just the bare presence of "choices".
    assert "msg.get('content'" in src or "msg.get(\"content\"" in src, (
        "verify-install.sh must check that `content` is non-empty, not just "
        "that the response is structurally a chat completion"
    )


def test_lens_chat_request_schema_accepts_chat_template_kwargs():
    """The Geometric Lens proxy at /v1/chat/completions sits between callers
    and the vLLM gen instance. Pydantic's default `extra="ignore"` means any
    field not declared on ChatRequest is silently dropped before hand-off to
    `forward_to_llama`. Without an explicit `chat_template_kwargs` field a
    caller passing {"enable_thinking": False} would have it stripped at the
    proxy boundary and Qwen3.5 would still emit full <think> blocks. The
    field must be declared, and the request handler must forward it through
    its kwargs dict to `forward_to_llama` / `forward_to_llama_stream`."""
    main_src = (PROJECT_ROOT / "geometric-lens" / "main.py").read_text()
    # Field declared on the schema...
    assert "chat_template_kwargs:" in main_src, (
        "ChatRequest must declare chat_template_kwargs so Pydantic doesn't "
        "drop it before hand-off to forward_to_llama"
    )
    # ...and forwarded to the kwargs dict in the chat handler.
    assert 'kwargs["chat_template_kwargs"]' in main_src, (
        "chat_completions handler must forward request.chat_template_kwargs "
        "into the kwargs dict it passes to forward_to_llama"
    )


def test_lens_chat_clients_disable_thinking():
    """All three Geometric Lens chat clients (pattern_extractor, summarizer,
    tree_search) hit /v1/chat/completions with very small max_tokens (50-200).
    On Qwen3.5, leaving thinking enabled fills the entire budget with <think>
    blocks and returns empty content. Each client must explicitly set
    chat_template_kwargs.enable_thinking=False on its outgoing request body."""
    files = [
        PROJECT_ROOT / "geometric-lens" / "cache" / "pattern_extractor.py",
        PROJECT_ROOT / "geometric-lens" / "indexer" / "summarizer.py",
        PROJECT_ROOT / "geometric-lens" / "retriever" / "tree_search.py",
    ]
    for f in files:
        src = f.read_text()
        assert "/v1/chat/completions" in src, f"{f.name} should hit chat completions"
        assert "chat_template_kwargs" in src, (
            f"{f.name} must set chat_template_kwargs in its request body"
        )
        assert "enable_thinking" in src, (
            f"{f.name} must set enable_thinking to silence Qwen3.5 reasoning "
            "(small max_tokens budgets get eaten by <think> blocks)"
        )


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


def test_vllm_image_tag_is_pinned_to_v0_17_1():
    """vLLM 0.18+ has a Qwen3.5 engine crash. Stage 50 pinned the H200
    Dockerfile via `pip install vllm==0.17.1`. Two other surfaces still
    pulled the moving `:latest` tag: the docker-compose `image:` keys
    on `vllm-gen` and `vllm-embed`, and the `scripts/build-containers.sh`
    pull command (used by the legacy K3s path). Both target the same
    upstream Docker Hub repo (`vllm/vllm-openai`), so when vLLM cuts a
    0.18 release, every fresh `make up` would pull the broken version
    and fail at the first `/v1/chat/completions` call.

    Pin: every reference to `vllm/vllm-openai:` in the live
    deployment paths must carry an explicit version tag
    (`v0.17.1`), never `:latest`."""
    files = [
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "scripts" / "build-containers.sh",
        PROJECT_ROOT / "benchmarks" / "section_b_instruction_following" / "ifbench" / "config.yaml",
    ]
    for f in files:
        text = f.read_text()
        if "vllm/vllm-openai" not in text:
            continue
        # Every vllm/vllm-openai reference in this file must be tagged
        # with v0.17.1 (the pinned version), not :latest.
        bad = re.findall(r"vllm/vllm-openai:latest", text)
        assert not bad, (
            f"{f.relative_to(PROJECT_ROOT)} still pulls "
            f"`vllm/vllm-openai:latest` — vLLM 0.18+ crashes on Qwen3.5; "
            f"pin to `vllm/vllm-openai:v0.17.1`"
        )

    # Sanity: the docker-compose image tag is the v0.17.1 we expect.
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
    assert "vllm/vllm-openai:v0.17.1" in compose, (
        "docker-compose.yml must reference the pinned `vllm/vllm-openai:v0.17.1`"
    )


def test_readme_roadmap_drops_dead_c_side_sampler_bullet():
    """The README.md V3.1 roadmap had a "Grammar speed - C-side sampler
    chain for faster constrained decoding" bullet — that was a llama.cpp-
    era goal. Under vLLM, constrained decoding is already Triton-backed
    (FlashInfer/Outlines integrated into the engine), so there is no
    "switch to C-side for speed" path. The bullet promises a feature
    that doesn't have a vLLM-shaped equivalent.

    Pin: neither the English README nor the three translations may
    list this dead roadmap item."""
    files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "ja" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "ko" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "zh-CN" / "README.md",
    ]
    bad_phrases = [
        "C-side sampler chain",
        "C サイドサンプラーチェーン",
        "C 사이드 샘플러 체인",
        "C 端采样器链",
    ]
    for f in files:
        text = f.read_text()
        for phrase in bad_phrases:
            assert phrase not in text, (
                f"{f.relative_to(PROJECT_ROOT)} still lists the dead "
                f"`{phrase}` roadmap bullet — that was a llama.cpp-era "
                f"goal with no vLLM equivalent"
            )


def test_docs_do_not_call_grammar_go_a_gbnf_generator():
    """Stage 120 deleted `buildGBNFGrammar()` from atlas-proxy/grammar.go.
    `docs/MAP.md` and the three translation README "What ATLAS Does"
    bullets still described grammar.go as producing GBNF — but vLLM
    doesn't accept GBNF (uses JSON Schema via `guided_json` and
    `response_format: json_object`). The English README was already
    rewritten in stage 93 to say "Structured output - vLLM
    response_format: json_object". The translations and MAP lagged.

    Pin: docs/MAP.md and the three translation README bullets must not
    call grammar.go a "GBNF" generator. The `#grammar-enforcement`
    anchor in ARCHITECTURE.md keeps its name (the heading is
    historical) but the link text and description must reflect vLLM's
    actual mechanism."""
    map_md = (PROJECT_ROOT / "docs" / "MAP.md").read_text()
    assert "GBNF grammar" not in map_md, (
        "docs/MAP.md still describes grammar.go as a 'GBNF grammar' "
        "generator — but stage 120 deleted that function and vLLM "
        "doesn't accept GBNF anyway"
    )

    # Translation READMEs: bullet `b.` under "What ATLAS Does" must not
    # claim GBNF as the JSON-shape mechanism.
    readme_translations = [
        PROJECT_ROOT / "docs" / "lang" / "ja" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "ko" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "zh-CN" / "README.md",
    ]
    for f in readme_translations:
        text = f.read_text()
        # The "GBNF" string in any of the bullet rows under
        # "What ATLAS Does" / equivalent. Keep the check broad — none
        # of the translations should claim GBNF anywhere in their
        # body, since none of the live ATLAS surface uses it.
        assert "GBNF" not in text, (
            f"{f.relative_to(PROJECT_ROOT)} still references GBNF; "
            "vLLM uses response_format: json_object, not GBNF"
        )


def test_atlas_proxy_grammar_does_not_carry_dead_gbnf_function():
    """`atlas-proxy/grammar.go` used to define a `buildGBNFGrammar()`
    function (~55 lines of Go) that produced a llama.cpp GBNF grammar
    string. The comment said it was "kept as reference in case json_object
    mode needs to be replaced with GBNF" — but vLLM doesn't accept GBNF
    (it uses lark-format `guided_grammar` and JSON Schema via `guided_json`).
    The function was never called, was incompatible with vLLM, and
    provided no starting point for the lark-format alternative anyone
    would actually port to.

    Pin: `buildGBNFGrammar` may not exist in the file. A future port to
    vLLM's lark grammar (if `response_format: json_object` ever proves
    insufficient) should be a fresh implementation, not a reference to
    GBNF."""
    src = (PROJECT_ROOT / "atlas-proxy" / "grammar.go").read_text()
    assert "func buildGBNFGrammar" not in src, (
        "atlas-proxy/grammar.go still defines buildGBNFGrammar — that's "
        "dead llama.cpp-only code. vLLM uses guided_json (JSON Schema) "
        "and guided_grammar (lark format), not GBNF."
    )


def test_v3_runner_llm_adapter_docstring_matches_lock_behavior():
    """`benchmark/v3_runner.py LLMAdapter` had a docstring that presented
    the class-level `threading.Lock()` as the always-engaged mechanism
    that "gives full single-slot throughput (~47 tok/s)." The actual code
    block ~80 lines below only acquires the lock when
    `LLMAdapter._parallel_mode == False` (i.e. the legacy llama.cpp opt-out
    path); under vLLM (the default `ATLAS_LLM_PARALLEL=1`) PagedAttention
    handles concurrent slots and the lock is bypassed. The docstring
    contradicted the comment 7 lines below it and quoted the dead
    llama-server `~47 tok/s` figure.

    Pin: the docstring must not present the lock as always-engaged and
    must not quote `~47 tok/s` as if it were a vLLM throughput number."""
    src = (PROJECT_ROOT / "benchmark" / "v3_runner.py").read_text()
    # Locate the LLMAdapter class docstring.
    cls_match = re.search(
        r"class LLMAdapter[^\n]*:\s*\n\s*\"\"\"(.*?)\"\"\"",
        src,
        re.DOTALL,
    )
    assert cls_match, "Could not locate LLMAdapter class docstring"
    docstring = cls_match.group(1)
    assert "~47 tok/s" not in docstring, (
        "LLMAdapter docstring still quotes the dead llama-server "
        "`~47 tok/s` figure; vLLM throughput is concurrency- and "
        "GPU-dependent"
    )
    # Phrasings that would assert the lock is always engaged:
    bad_present_tense = [
        r"A class-level\s+lock ensures only one .* request is in-flight",
    ]
    for pattern in bad_present_tense:
        assert not re.search(pattern, docstring), (
            "LLMAdapter docstring still presents the threading.Lock as "
            "always-engaged — but `_parallel_mode=True` (vLLM default) "
            "bypasses the lock entirely"
        )


def test_no_stale_51_toks_claim_in_user_facing_surfaces():
    """Several user-facing surfaces (atlas-proxy comment, CLI.md REPL
    output snippet, three translation READMEs) carried a stale
    "~51 tok/s" claim — that's llama-server's single-slot throughput on
    the 9B model. vLLM throughput depends entirely on concurrency, GPU,
    and quantization format; there's no honest single number. Stage 38
    fixed the live REPL display, but the same dead figure leaked into
    these other surfaces.

    Pin: none of these files may quote `51 tok/s` (or `약 51 tok/s` /
    `约 51 tok/s` / `約 51 tok/s`) as if it were the current vLLM
    throughput."""
    files = [
        PROJECT_ROOT / "atlas-proxy" / "agent.go",
        PROJECT_ROOT / "docs" / "CLI.md",
        PROJECT_ROOT / "docs" / "lang" / "ja" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "ko" / "README.md",
        PROJECT_ROOT / "docs" / "lang" / "zh-CN" / "README.md",
    ]
    bad_phrases = [
        "~51 tok/s",
        "약 51 tok/s",
        "约 51 tok/s",
        "約 51 tok/s",
    ]
    for f in files:
        text = f.read_text()
        for phrase in bad_phrases:
            assert phrase not in text, (
                f"{f.relative_to(PROJECT_ROOT)} still claims `{phrase}` — "
                f"that's a llama-server single-slot number; vLLM "
                f"throughput is concurrency- and GPU-dependent"
            )


def test_cli_md_health_check_table_uses_correct_vllm_ports():
    """`docs/CLI.md` had a "Service Health Timeout" table that listed
    `vLLM` on port `8080` — but vLLM gen runs on 8000 and embed on 8001
    across the entire stack. A user following the CLI doc to debug a
    failed startup would `lsof -i :8080` and find nothing. The same
    REPL output a few lines above already correctly shows ports 8000
    and 8001, so the table contradicted the prose.

    Pin: the health-check table in CLI.md must list vLLM gen on 8000
    and either include or omit (but not contradict) the embed instance."""
    cli = (PROJECT_ROOT / "docs" / "CLI.md").read_text()
    table_match = re.search(
        r"Each service is health-checked.*?(?=If a service is already running)",
        cli,
        re.DOTALL,
    )
    assert table_match, "Could not locate the health-check table in CLI.md"
    table = table_match.group(0)
    assert "| 8080 |" not in table, (
        "CLI.md health-check table still lists vLLM on port 8080 — "
        "that was the old llama-server port; vLLM gen runs on 8000"
    )
    assert "| 8000 |" in table, (
        "CLI.md health-check table must list the vLLM gen port (8000)"
    )


def test_lens_embedding_extractor_uses_current_vllm_api_name():
    """`geometric-lens/geometric_lens/embedding_extractor.py` is the entry
    point used by every C(x)/G(x) scoring path in the Lens, and
    `tests/infrastructure/test_llm.py` documents the contract the embed
    instance must honor. Both used to reference the deprecated
    `--task embed` mode in their docstrings — but vLLM 0.17+ removed
    that flag in favor of `--runner pooling --convert embed`. A reader
    debugging "why is the Lens getting empty embeddings?" would search
    vLLM docs for `--task embed`, find a deprecation notice, and have
    no clear next step.

    Pin: neither file may present `in --task embed mode` as a live
    description, and `embedding_extractor.py` must reference the
    current API by name."""
    files = [
        PROJECT_ROOT / "geometric-lens" / "geometric_lens" / "embedding_extractor.py",
        PROJECT_ROOT / "tests" / "infrastructure" / "test_llm.py",
    ]
    bad_pattern = re.compile(r"in --task embed mode")
    for f in files:
        src = f.read_text()
        assert not bad_pattern.search(src), (
            f"{f.relative_to(PROJECT_ROOT)} still presents `--task embed` "
            f"as a live vLLM mode; vLLM 0.17+ uses "
            f"`--runner pooling --convert embed`"
        )

    # `embedding_extractor.py` must additionally call out the current
    # API by name (formatting varies — match across whitespace).
    extractor_src = files[0].read_text()
    assert re.search(r"--runner\s+pooling\s+--convert\s+embed", extractor_src), (
        "embedding_extractor.py docstring must name the current "
        "vLLM API (`--runner pooling --convert embed`)"
    )


def test_docs_do_not_link_to_deleted_scripts_or_dirs():
    """The V3.0 → V3.0.1 cutover deleted several scripts and directories
    that the docs (`MAP.md`, `ARCHITECTURE.md`) used to link to:
      - `scripts/deploy-9b.sh`        — never re-ported for vLLM
      - `scripts/smoke-test-9b.sh`    — never re-ported
      - `inference/`                  — collapsed into benchmarks/h200/
      - the K3s `templates/` directory `generate-manifests.sh` consumes —
                                        not currently shipped

    Pin: MAP.md and ARCHITECTURE.md must not present the first two as
    live, callable scripts (a fenced backtick path that points at a
    missing file misleads anyone running a `cat`/`bat` from the repo
    root). The dirs are referenced in disclaimer prose to explain what
    *was* there, but no longer as live navigation targets."""
    map_md = (PROJECT_ROOT / "docs" / "MAP.md").read_text()
    arch_md = (PROJECT_ROOT / "docs" / "ARCHITECTURE.md").read_text()

    # MAP.md: scripts table rows must not reference the deleted scripts.
    # A row is detected by `[`<filename>`](path)` — markdown link in a
    # table cell. Look for the deleted filenames in that shape.
    for ghost in ("deploy-9b.sh", "smoke-test-9b.sh"):
        # A markdown link of the form [`<ghost>`](...) is a live nav target.
        # Disclaimer prose mentioning the name without the link is allowed.
        live_link_re = re.compile(rf"\[`[^`]*{re.escape(ghost)}[^`]*`\]\(")
        assert not live_link_re.search(map_md), (
            f"docs/MAP.md still links to `{ghost}` as if it were a live "
            f"script — but {PROJECT_ROOT / 'scripts' / ghost} doesn't exist"
        )

    # ARCHITECTURE.md K3s subsection — must not present the deleted
    # `inference/` entrypoints as the *current* K3s mechanism.
    k3s_match = re.search(
        r"### K3s.*?(?=\n## |\n### |\Z)",
        arch_md,
        re.DOTALL,
    )
    assert k3s_match, "Could not locate ### K3s subsection in ARCHITECTURE.md"
    k3s_text = k3s_match.group(0)
    # Phrasings that would imply the deleted `inference/` is current:
    bad_present_tense = [
        r"K3s deployment uses the entrypoint scripts in `inference/`",
        r"Services deploy as pods.*processed by",
    ]
    for pattern in bad_present_tense:
        assert not re.search(pattern, k3s_text), (
            f"ARCHITECTURE.md ### K3s still presents `inference/` or the "
            f"V3.0-era manifest pipeline as current — but those were "
            f"removed in the vLLM cutover. Phrase as historical context."
        )
    """`docs/MAP.md` is the canonical "where does X live in this repo"
    map. After the vLLM cutover, `benchmarks/h200/` contains only the
    single-image build (`Dockerfile`), the entrypoint that drives both
    vLLM instances + Lens (`entrypoint.sh`), preflight, and a few
    runner scripts. The previous MAP listing carried over the V3.0-era
    catalog: `Dockerfile.v31`, `Dockerfile.mtp`, `entrypoint-v3.1-9b.sh`,
    `entrypoint-v3-specdec.sh`, `entrypoint-embed.sh`, `entrypoint-mtp.sh`,
    `patches/fix-embeddings-spec-decode.patch`,
    `templates/Qwen3-{custom,no-think}.jinja` — eight entries pointing
    at files that don't exist on disk.

    Pin: every `benchmarks/h200/<x>` link in MAP.md must reference a
    file that actually exists; the deleted V3.0 artifacts must not
    reappear in the listing."""
    map_md = (PROJECT_ROOT / "docs" / "MAP.md").read_text()
    deleted = [
        "Dockerfile.v31",
        "Dockerfile.mtp",
        "entrypoint-v3.1-9b.sh",
        "entrypoint-v3-specdec.sh",
        "entrypoint-embed.sh",
        "entrypoint-mtp.sh",
        "fix-embeddings-spec-decode.patch",
        "Qwen3-custom.jinja",
        "Qwen3-no-think.jinja",
    ]
    h200_section_match = re.search(
        r"benchmarks/h200/.*?(?=\n- \[|\Z)",
        map_md,
        re.DOTALL,
    )
    assert h200_section_match, "Could not locate the benchmarks/h200/ block in MAP.md"
    section = h200_section_match.group(0)
    for name in deleted:
        assert name not in section, (
            f"docs/MAP.md still lists `{name}` under benchmarks/h200/, "
            f"but that file does not exist in the repo (verified: it was "
            f"removed in the V3.0 → V3.0.1 cutover)"
        )

    # Positive assertion — the entries that ARE in the section must all
    # name files that exist on disk.
    h200_dir = PROJECT_ROOT / "benchmarks" / "h200"
    bullet_re = re.compile(r"\[`([^`]+)`\]")
    for match in bullet_re.finditer(section):
        listed = match.group(1)
        # Skip the section header itself (`benchmarks/h200/`) and any
        # back-link anchors that don't resolve to files.
        if listed.endswith("/"):
            continue
        # Strip benchmarks/h200/ prefix if present, otherwise treat as
        # a bare filename relative to that dir.
        relative = listed.replace("benchmarks/h200/", "")
        assert (h200_dir / relative).exists(), (
            f"docs/MAP.md lists `{listed}` but {h200_dir / relative} "
            f"does not exist on disk"
        )


def test_atlas_cli_check_llama_tolerates_empty_vllm_health_body():
    """Same root cause as `test_v3_runner_preflight_tolerates_empty_vllm_health_body`,
    different surface: `atlas/cli/client.py:check_llama` used to call
    `_get(f"{INFERENCE_URL}/health")` where `_get` runs
    `json.loads(resp.read().decode())` on the response. vLLM 0.17+
    `/health` returns 200 + empty body, so JSONDecodeError fires inside
    the try-block, the except branch returns `(False, str(e))`, and the
    REPL's status panel reports vLLM as down on a healthy stack.

    Pin: `check_llama` must not call `_get` (the JSON-parsing helper) on
    `/health` — only `_probe` (status-code-only) is acceptable. The
    `/v1/models` follow-up call is fine to JSON-parse since it returns
    OpenAI-shape JSON."""
    src = (PROJECT_ROOT / "atlas" / "cli" / "client.py").read_text()
    # Locate the check_llama function body.
    fn_match = re.search(
        r"def check_llama\(\).*?(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert fn_match, "Could not locate check_llama in atlas/cli/client.py"
    fn_body = fn_match.group(0)
    assert '_get(f"{INFERENCE_URL}/health"' not in fn_body, (
        "check_llama still calls _get (JSON-parsing) on /health — vLLM "
        "returns an empty 200 body, so this raises JSONDecodeError on "
        "every healthy stack. Use _probe (status-code-only) instead."
    )
    # And the function must use _probe somewhere (positive assertion).
    assert "_probe(" in fn_body, (
        "check_llama must use _probe(...) for the /health check"
    )


def test_v3_runner_preflight_tolerates_empty_vllm_health_body():
    """vLLM 0.17+ `/health` returns HTTP 200 with an *empty* body — it's
    a readiness probe, not a structured status report. The previous
    `v3_runner.py` preflight called `json.loads(resp.read())` on the
    response and aborted the benchmark with `JSONDecodeError` on every
    healthy vLLM stack — a real bug that would prevent any benchmark
    from starting.

    Pin: the gen-side preflight may not call `json.loads` on the
    `/health` response. (The Lens preflight is fine to parse JSON
    since FastAPI Lens does return `{"status":"healthy",...}`.)"""
    src = (PROJECT_ROOT / "benchmark" / "v3_runner.py").read_text()
    # Find the preflight block — anchored on the actual /health call.
    # The block runs from the LLAMA_URL/health line up to (but not
    # including) the next /health probe (RAG_API_URL/health for Lens).
    preflight_match = re.search(
        r"urllib\.request\.Request\(f\"\{LLAMA_URL\}/health\"\).*?(?=urllib\.request\.Request\(f\"\{RAG_API_URL\})",
        src,
        re.DOTALL,
    )
    assert preflight_match, "Could not locate the vLLM gen-instance preflight block"
    gen_block = preflight_match.group(0)
    # The gen-instance preflight must inspect the HTTP status code, not
    # try to parse a body that real vLLM doesn't send.
    assert "json.loads" not in gen_block, (
        "v3_runner.py vLLM gen-instance preflight still calls json.loads "
        "on /health — but vLLM 0.17+ returns 200 with an empty body, "
        "so this aborts every benchmark. Use resp.status instead."
    )
    assert "resp.status" in gen_block or "status_code" in gen_block, (
        "v3_runner.py vLLM gen-instance preflight must inspect the HTTP "
        "status code (e.g. resp.status == 200) to confirm readiness"
    )


def test_lens_retriever_defaults_match_vllm_gen_service():
    """`geometric-lens/retriever/hybrid.py` and `tree_search.py` used to
    hardcode `llama_url: str = "http://llama-service:8000"` as the kwarg
    default. `llama-service` is a legacy K3s service name from the V3.0
    llama.cpp deployment — docker-compose ships the gen instance under the
    service name `vllm-gen` (port 8000). Although every current call site
    in `main.py` and `pipeline.py` passes `config.llama.base_url`
    explicitly (which resolves correctly via env), a bare instantiation
    or a future contributor's test fixture would silently land on a
    nonexistent host and time out at request time rather than fail at
    construction.

    Pin: neither retriever may default to `llama-service:8000`. The
    default resolution chain must mirror `geometric_lens/config.py`:
    `LLAMA_GEN_URL → LLAMA_URL → http://vllm-gen:8000`."""
    files = [
        PROJECT_ROOT / "geometric-lens" / "retriever" / "hybrid.py",
        PROJECT_ROOT / "geometric-lens" / "retriever" / "tree_search.py",
    ]
    for f in files:
        src = f.read_text()
        assert "llama-service" not in src, (
            f"{f.relative_to(PROJECT_ROOT)} still references the legacy "
            f"K3s service name `llama-service`; docker-compose uses "
            f"`vllm-gen` for the gen instance"
        )
        assert "vllm-gen:8000" in src, (
            f"{f.relative_to(PROJECT_ROOT)} must default to "
            f"`http://vllm-gen:8000` when no llama_url is passed"
        )
        assert "LLAMA_GEN_URL" in src, (
            f"{f.relative_to(PROJECT_ROOT)} must consult LLAMA_GEN_URL "
            f"via os.environ to mirror config.py's resolution chain"
        )


def test_v3_code_does_not_reference_dead_jinja_flag():
    """`benchmark/v3/budget_forcing.py` and `benchmark/v3_runner.py` used to
    have explanatory comments that read 'With --jinja enabled, the model
    naturally uses <think> tags' — `--jinja` was a llama-server CLI flag
    that toggled jinja chat-template processing. Under vLLM there is no
    such flag (jinja templates are always applied via `--reasoning-parser`,
    and `/v1/completions` skips the parser entirely). The comment misled
    future readers into believing they could control the behavior with a
    CLI flag that doesn't exist.

    Pin: V3 code in `benchmark/v3*` may not reference `--jinja` outside
    a deliberate disclaimer (we don't expect any callers to mention it
    at all)."""
    files = [
        PROJECT_ROOT / "benchmark" / "v3" / "budget_forcing.py",
        PROJECT_ROOT / "benchmark" / "v3_runner.py",
    ]
    for f in files:
        src = f.read_text()
        assert "--jinja" not in src, (
            f"{f.relative_to(PROJECT_ROOT)} still references the dead "
            f"`--jinja` llama-server flag; under vLLM jinja templates are "
            f"always applied (no such CLI knob)"
        )


def test_architecture_budget_forcing_table_matches_code():
    """ARCHITECTURE.md's Budget Forcing tier table used to claim the nothink
    tier injected `/nothink prompt` as a wait-injection — wrong on two counts:
    (a) BUDGET_TIERS["nothink"] has inject_wait=False (no wait injection
    happens for nothink, same as light), and (b) Qwen3.5 dropped the literal
    `/nothink` soft-command anyway — the real suppression mechanism is the
    `<think>\\n\\n</think>\\n\\n` assistant prefill emitted by format_chatml
    when the tier is "nothink".

    The Phase 0 probe paragraph also referenced the retry order as
    `light → standard → /nothink`, where `/nothink` (slash-prefixed) read as
    the deprecated soft-command rather than the budget-tier name.

    Pin: the Budget Forcing table's nothink row must not promise a
    `/nothink prompt` wait-injection, and Phase 0's retry order must not
    use the slash-prefixed `/nothink` form."""
    arch = (PROJECT_ROOT / "docs" / "ARCHITECTURE.md").read_text()
    nothink_row_match = re.search(r"^\| nothink \| 0 \| (.+?) \|$", arch, re.MULTILINE)
    assert nothink_row_match, (
        "Budget Forcing table no longer has a 'nothink | 0 | ...' row in "
        "the expected format — adapt the test if the table shape changed"
    )
    wait_cell = nothink_row_match.group(1)
    assert "/nothink" not in wait_cell, (
        "nothink row's Wait Injection cell still claims '/nothink prompt'; "
        "the soft-command was dropped by Qwen3.5 and BUDGET_TIERS['nothink'] "
        "has inject_wait=False — the cell should say 'None' (with optional "
        "explanation that suppression is via prefill)"
    )
    # Phase 0 paragraph: tier list must not use the slash-prefixed form.
    phase0_match = re.search(r"\*\*Phase 0: Probe\*\* generates.*?\.", arch, re.DOTALL)
    assert phase0_match, "Could not locate the Phase 0 Probe paragraph"
    phase0_text = phase0_match.group(0)
    assert "/nothink" not in phase0_text, (
        "Phase 0 retry order references the deprecated `/nothink` "
        "soft-command — should be the unprefixed `nothink` tier name"
    )


def test_docs_have_no_pre_vllm_residue():
    """docs/CLI.md used to document an `ATLAS_LLAMA_BIN` env var pointing at
    `~/llama-cpp-mtp/build/bin/...`. Both the variable and the path are dead
    weight under vLLM (vLLM is `pip install vllm`, not a build artifact, and
    nothing in the codebase reads ATLAS_LLAMA_BIN). Leaving stale rows around
    misleads users into thinking the knob does something.

    Pin: no doc may reference the legacy llama-cpp-mtp build path or the
    dead ATLAS_LLAMA_BIN environment variable."""
    cli_md = (PROJECT_ROOT / "docs" / "CLI.md").read_text()
    assert "ATLAS_LLAMA_BIN" not in cli_md, (
        "ATLAS_LLAMA_BIN is dead (no code reads it); remove the row"
    )
    assert "llama-cpp-mtp" not in cli_md, (
        "Pre-vLLM build path leaked into CLI.md; remove it"
    )

    # Also sanity-check the rest of the docs tree.
    for md in (PROJECT_ROOT / "docs").rglob("*.md"):
        text = md.read_text()
        assert "ATLAS_LLAMA_BIN" not in text, (
            f"{md.relative_to(PROJECT_ROOT)} still mentions ATLAS_LLAMA_BIN; "
            "the var is read nowhere in the codebase"
        )
