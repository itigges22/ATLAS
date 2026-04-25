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
    """`docs/TROUBLESHOOTING.md`'s Performance section steered users to
    debug `--n-gpu-layers 99`, a llama.cpp flag with no vLLM equivalent
    (vLLM auto-offloads every layer when CUDA is reachable). Quoting
    llama.cpp flags as troubleshooting steps for a vLLM stack misleads
    users into chasing a knob that doesn't exist.

    Forbid the llama.cpp-specific `--n-gpu-layers` flag, except inside an
    explanation of why it doesn't apply (a "no `--n-gpu-layers` knob"
    explainer is the *fix*). Same for the obsolete "grammar enforcement"
    throughput claim."""
    src = (PROJECT_ROOT / "docs" / "TROUBLESHOOTING.md").read_text()
    # Allow the explanatory pattern that explicitly disclaims the flag.
    body = src.replace("there is no `--n-gpu-layers` knob", "")
    assert "--n-gpu-layers" not in body, (
        "TROUBLESHOOTING.md must not steer users to debug --n-gpu-layers; "
        "it's a llama.cpp flag with no vLLM analogue"
    )
    assert "with grammar enforcement" not in src, (
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
    """The atlas-proxy has three places that POST to /v1/chat/completions:
    `forwardToFox` (the canonical helper), the spec generator, and the
    streaming-handler fallback. Earlier iterations only set
    enable_thinking=false in forwardToFox — the other two paths shipped
    Qwen3.5 requests with thinking enabled, then stripped <think> blocks
    after the fact (or, in streaming mode, left them in the response).
    The helper `applyVllmDefaults` must be applied to every direct POST
    so all three paths agree on the request shape."""
    src = (PROJECT_ROOT / "atlas-proxy" / "main.go").read_text()
    # Helper must exist.
    assert "func applyVllmDefaults(req *ChatRequest)" in src, (
        "atlas-proxy must define an applyVllmDefaults helper to ensure"
        " every chat POST sends enable_thinking=false consistently"
    )
    # Helper must set the kwarg.
    helper_start = src.index("func applyVllmDefaults")
    helper_end = src.index("\n}", helper_start)
    helper_body = src[helper_start:helper_end]
    assert "enable_thinking" in helper_body, (
        "applyVllmDefaults must set chat_template_kwargs.enable_thinking"
    )
    # Count callers — must be at least 3 (forwardToFox + spec + streaming).
    apply_calls = src.count("applyVllmDefaults(&")
    assert apply_calls >= 3, (
        f"applyVllmDefaults should be called from forwardToFox + spec generator "
        f"+ streaming fallback (3 sites), found {apply_calls}"
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
