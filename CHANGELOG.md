# Changelog

## Unreleased

### vLLM Migration (replaces llama.cpp / llama-server stack)
- Inference backend ported from llama.cpp to vLLM 0.17+ (Qwen3.5 DeltaNet hybrid kernels via Triton).
- **Two vLLM instances** — vLLM serves only one task per process, so generation runs on port 8000 (`--reasoning-parser qwen3 --enable-prefix-caching`) and embeddings on port 8001 (`--runner pooling --convert embed` returning 4096-dim hidden states for the Geometric Lens). Same model weights, two processes.
- **Model**: switched from GGUF Q6_K to AWQ Q4 (`QuantTrio/Qwen3.5-9B-AWQ`, ~12 GiB) since vLLM doesn't load GGUF natively.
- **Endpoints**: dropped llama.cpp-only `/completion` and `/embedding`; everything now uses `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` (OpenAI-compatible).
- **`/nothink` injection removed**: Qwen3.5 dropped the soft `/think` and `/nothink` commands. Thinking is now controlled via the request body field `chat_template_kwargs.enable_thinking`.
- **`inference/` directory deleted entirely**: 4 Dockerfiles, 6 entrypoints, jinja templates, and the llama.cpp embeddings/spec-decode patch all removed. The single source of truth is now `benchmarks/h200/{Dockerfile,entrypoint.sh,preflight.sh}`.
- **Pre-flight script** (`benchmarks/h200/preflight.sh`): hits gen + embed + Lens with real requests before the benchmark sweep starts. Refuses to run if any service is misconfigured.
- **Wire test suite** (`tests/integration/test_vllm_wire.py`): 11 tests covering chat completions, raw completions with token_logprobs, thinking on/off, reasoning_content split, missing-parser fallback, missing-logprobs fallback, empty-content (budget exhausted), 503 retry, embeddings (single + batch), and the legacy `cache_prompt` kwarg back-compat path.
- **Docker-compose invariant tests** (`tests/integration/test_docker_compose_invariants.py`): 10 tests that simulate `${VAR:-default}` substitution against `.env.example` and verify the resolved config still satisfies the runtime invariants — gen on 8000 with `--reasoning-parser qwen3`, embed on 8001 with `--runner pooling --convert embed` (NOT deprecated `--task embed`), Lens healthcheck targets 31144, atlas-proxy + v3-service env vars use docker DNS names, HF_TOKEN threaded to both vLLM instances, no legacy llama-server service, no `.gguf` paths.
- **CI workflow** (`.github/workflows/vllm-wire.yml`): runs the wire tests + invariant tests + 854 unit tests + docker-compose YAML validation + `bash -n` on every entrypoint script on every push and PR. Uses `pip install --no-deps -e .` to skip aider-chat's transitive deps (cuts CI install time ~80%).
- **Makefile**: one-command access to the verification surface — `make test`, `make wire-tests`, `make lint` (11 files), `make up`/`down`/`logs` (docker-compose lifecycle), `make model` (huggingface-cli pull), `make preflight`.
- **Lens chat clients fixed**: `pattern_extractor.py`, `summarizer.py`, `tree_search.py` all sent `model="default"` which vLLM rejects. Now read `LLAMA_GEN_MODEL` (default `qwen3.5-9b`).
- **Config split**: `LLAMA_URL` is split into `LLAMA_GEN_URL` and `LLAMA_EMBED_URL`; old name stays as fallback for backwards compat. `ATLAS_INFERENCE_URL` → 8000 not 8080.
- **Lens port aligned to 31144**: container, docker-compose, atlas.conf.example, benchmark configs, and all docs now use the same port (was 8099 in some places).
- **HF auth**: docker-compose threads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` through to both vLLM instances for gated/private repos.
- **`.env.example`** rewritten with the new vLLM knobs (`ATLAS_GEN_*`, `ATLAS_EMBED_*`).
- Install/uninstall/verify scripts now check for the AWQ model directory instead of `*.gguf` files.
- **AWQ download helper** (`scripts/download-models.sh`): rewritten to pull `QuantTrio/Qwen3.5-9B-AWQ` via `huggingface-cli` instead of curl-fetching GGUF shards. Honors `HF_TOKEN` for gated/private repos.
- **Transformers patch is content-based**: the `partial_rotary_factor` patch in the Dockerfile no longer edits a hardcoded line number — it finds the existing `ignore_keys_at_rope_validation = ...` assignment via regex and injects after it. Future transformers refactors that re-arrange the file but keep the anchor will still patch correctly; ones that remove the anchor fail loudly at build time.
- **Hardcoded model name strings removed**: 3 occurrences in `benchmarks/v301_runner.py` result JSON output, 1 in `v3-service/main.py` fallback, the `v2_report.py` header text, and `transfer_to_h200.sh` rsync source — all updated to use `LLAMA_GEN_MODEL` (`qwen3.5-9b`) instead of `Qwen3.5-9B-Q6_K`.
- **REPL launcher** (`atlas/cli/repl.py`): `_launch_local_proxy` now exports `LLAMA_GEN_URL`, `LLAMA_EMBED_URL`, `LLAMA_GEN_MODEL`, `LLAMA_EMBED_MODEL` to the proxy subprocess (was missing the embed half — proxy fell back to compiled-in default).
- **`benchmarks/h200/launch_on_h200.sh`** rewritten for vLLM: builds `benchmarks/h200/Dockerfile` (deleted: building llama.cpp from source), starts the container with `MODE` + `DOWNLOAD_MODEL` env, exposes 8000/8001/31144 instead of 32735.
- **`benchmarks/h200/transfer_to_h200.sh`** rewritten for AWQ: rsyncs the `models/Qwen3.5-9B-AWQ/` directory + `geometric-lens/` tree (was a single `.gguf` file).
- Documentation: README intro + Known Limitations + Roadmap, MAP.md section list, all 7 active English docs, all 9 translation files (ja/ko/zh-CN), atlas-proxy/README.md, `benchmarks/{,h200/}README.md`, and the `tests/infrastructure/test_llm.py` docstrings — all updated for vLLM.
- **`.gitignore`**: excludes the new `.cache/huggingface/` mount path docker-compose creates by default.
- **Lens runtime deps**: `xgboost>=2.0` and `scikit-learn>=1.3` added to `geometric-lens/requirements.txt`. Without xgboost the Lens silently disabled G(x) candidate scoring at startup; the docker-compose Lens container shipped without it. Added two regression tests so this can't regress.
- **vLLM API up-to-date**: embed instance migrated from deprecated `--task embed` to `--runner pooling --convert embed` (the documented current API in vLLM 0.17+).
- **Pre-flight contract tests** verify the curl bodies in `benchmarks/h200/preflight.sh` match what the runners send: uses `chat_template_kwargs` (no `/nothink`), checks 4096-dim embed responses, hits OpenAI-compatible endpoints only.
- **Empty-content + missing-logprobs wire tests**: cover the two graceful-degradation paths the V3 adapter has to handle on real vLLM responses.
- **`atlas.conf.example` / `CONFIGURATION.md` / atlas-proxy README / benchmarks READMEs / TROUBLESHOOTING / methodology docs / AGGREGATE_REPORT**: all swept for the cutover. `ATLAS_LLAMA_NODEPORT` aliased to the new `ATLAS_VLLM_GEN_NODEPORT`; code paths now read the new name first, falling back to old. The TROUBLESHOOTING vLLM section was actively misleading (gave llama.cpp advice for a vLLM stack) — rewritten with concrete vLLM-correct fixes plus entries for the G(x) xgboost-missing path and the deprecated `--task embed` path.
- **`benchmark/measure_bok_latency.sh`**: default URL `localhost:32735` (old NodePort) → `LLAMA_GEN_URL` chain → `localhost:8000`.
- **`atlas/cli/repl.py`**: status block speed string `"~51 tok/s"` (llama.cpp single-slot rate) → `"vLLM (PagedAttention)"` since vLLM per-slot throughput varies wildly with concurrency/GPU/quant — no honest single number to display.
- **`.dockerignore` cleanup**: dropped dead `inference/` and `rag-api/` exclusions; removed an actively-misleading `geometric-lens/` exclusion in `benchmarks/h200/.dockerignore` (the Dockerfile COPYs that directory in).
- **Cloud-pod entrypoint `--served-model-name` flows from env**: previously hardcoded to `qwen3.5-9b` / `qwen3.5-9b-embed`, while preflight.sh and the Lens chat clients read `$LLAMA_GEN_MODEL` / `$LLAMA_EMBED_MODEL`. A user customizing the served name via `docker run -e LLAMA_GEN_MODEL=...` would have vLLM serving under the hardcoded name while every consumer asked for the customized one (vLLM rejects unknown names with a 4xx). Now both `vllm serve --served-model-name` invocations interpolate the env var, the entrypoint exports it (plus `LENS_URL`) before invoking preflight + runners, and the Dockerfile sets explicit defaults. Two new invariant tests pin this in place.
- **`benchmark/measure_bok_latency.sh` model name from env**: hardcoded `qwen3.5-9b` in three curl bodies. Now reads `${LLAMA_GEN_MODEL:-qwen3.5-9b}` once into `MODEL_NAME` and interpolates it into every payload — matches the convention every Python consumer (runner, v3_runner, geo_learning, Lens clients, REPL) already follows.
- **`atlas-proxy` thinking control via `chat_template_kwargs`**: Qwen3.5 dropped the `/think` and `/nothink` soft-commands; the proxy was still injecting `/nothink\n` into outgoing user/system prompts, where vLLM tokenized it as plain text and emitted full `<think>` blocks the proxy then stripped — wasting hundreds of generation tokens per call. Now `forwardToFox` sets `chat_template_kwargs.enable_thinking=false` on every outgoing request (callers can override with `true`), and the dead `/nothink` injection sites in `agent.go` + the multi-file generation paths in `main.go` were removed. Defensive `TrimPrefix` on inbound content remains so the proxy still tolerates upstream callers that include the prefix.
- **`v3-service` LLMAdapter forwards top_k/top_p/stop**: the adapter built an internal `body` dict with `top_k=20`, `top_p=0.95`, `stop=["\n\n\n\n"]`, then `_send` rebuilt `chat_body` for `/v1/chat/completions` *without* copying any of those params. Under llama.cpp the same fields had survived because the adapter sent the body unmodified; under vLLM they fell back to the model's `generation_config.json` defaults (Qwen3.5: `top_k=-1`, no stop), changing sampling behavior the V3 modules were tuned against. Now propagated explicitly. New wire test (`test_v3_service_adapter_propagates_sampling_params`) pins this in the chat completions payload shape.
- **Lens chat clients disable thinking**: `pattern_extractor`, `summarizer`, and `tree_search` all hit `/v1/chat/completions` with very small `max_tokens` (50-200) but never set `chat_template_kwargs.enable_thinking=false`. On Qwen3.5 this silently filled the entire budget with `<think>` blocks and returned empty content — pattern extraction failed silently, code summaries fell back to placeholders, and the JSON-only relevance scorer in `tree_search` returned truncated thinking text the parser couldn't interpret. All three now set the kwarg explicitly. New invariant test (`test_lens_chat_clients_disable_thinking`) reads each file and asserts the kwarg is present.
- **Lens `/v1/chat/completions` proxy forwards `chat_template_kwargs`**: the Geometric Lens FastAPI service exposes its own OpenAI-compatible chat completions endpoint that proxies to vLLM (with optional RAG enhancement). The Pydantic `ChatRequest` schema declared no `chat_template_kwargs` field, so any caller passing `{"enable_thinking": false}` had it dropped at the proxy boundary by Pydantic's default `extra="ignore"` — Qwen3.5 then emitted full `<think>` blocks the proxy faithfully streamed back. Field is now declared as `Optional[Dict[str, Any]]` and forwarded into the kwargs dict the handler passes to `forward_to_llama` / `forward_to_llama_stream`. Pinned by a new invariant test.
- **`scripts/verify-install.sh` LLM smoke check: real request, real validation**: the post-install verifier sent `/v1/chat/completions` with no `model` field (vLLM rejects unknown/missing names with a 4xx) and no `enable_thinking=false` (Qwen3.5 burned the entire `max_tokens=10` budget on a `<think>` block, returning empty content). It then only checked that the response contained `"choices"` — both failure modes returned structurally valid chat-completion shells with empty content, so the verifier falsely passed on a broken stack. Now sends `model: ${LLAMA_GEN_MODEL:-qwen3.5-9b}` + `chat_template_kwargs.enable_thinking=false`, raises `max_tokens` to 16, and parses the response with python to assert non-empty content. Pinned by a new invariant test.
- **atlas-proxy `applyVllmDefaults` helper covers all 3 chat-completion POSTs**: stage 65 set `enable_thinking=false` only inside `forwardToFox`. Two other paths POSTed directly to `/v1/chat/completions` and bypassed it — the spec generator (`MaxTokens=400`, would have spent the entire budget on `<think>`) and the streaming-handler fallback (any client message that didn't go through the buffered path). Extracted a shared `applyVllmDefaults(req *ChatRequest)` helper that pins the model name and disables thinking unless the caller opted in, and applied it at all three call sites. Pinned by a new invariant test that counts `applyVllmDefaults(&` occurrences.
- **v3-service image: dropped unused torch dep**: the Dockerfile installed the ~2 GiB torch CPU wheel — but neither `v3-service/main.py` (a stdlib `http.server` front-end) nor any of the `benchmark/v3/*` modules it imports actually uses torch, numpy, or any ML lib. Confirmed by replacing `torch` with a `_NoTorch` import-error stub and observing `import main` still succeeds. Build is now minutes faster and the image is much smaller. New invariant test forbids `pip install` lines that pull `torch`/`numpy`/`scipy`/`tensorflow` so the bloat can't sneak back in unnoticed.
- **`scripts/lib/config.sh validate_config` checks both vLLM NodePorts**: prior versions only listed `ATLAS_LLAMA_NODEPORT` in the dedup-and-range check. The new `ATLAS_VLLM_EMBED_NODEPORT` (defaulting to 32736) was unchecked — a fresh K8s install with the embed port colliding with another service's NodePort, or set outside the 30000-32767 K8s range, would slip through bash-side validation and only fail at `kubectl apply` time as an opaque "invalid value" error. Now `ATLAS_VLLM_GEN_NODEPORT` (with `ATLAS_LLAMA_NODEPORT` legacy fallback) and `ATLAS_VLLM_EMBED_NODEPORT` are both included. Pinned by a behavioral test that sources `config.sh`, sets a deliberately colliding embed/gen pair, and asserts validation fails.
- **`scripts/install.sh` short-circuits when the K8s manifests directory is missing**: the script's `deploy_manifests` step `kubectl apply`s `$K8S_DIR/manifests/{redis,llama,geometric-lens,llm-proxy,sandbox,training-cronjob}-deployment.yaml`, but the repo doesn't currently ship a `manifests/` directory at all — the canonical install path for the vLLM two-instance stack is docker-compose. Running `install.sh` blew up at the first `kubectl apply` with "no such file" and left the namespace half-initialized (Redis pulled, nothing else). `deploy_manifests` now checks for the directory's existence, prints a clear message pointing the user to `make up` / `make model` / `make preflight`, and exits with rc=2. Pinned by an invariant test that asserts both the guard and the docker-compose pointer message — the test also self-checks that `manifests/` really is absent, so a future PR that ports K8s support over will flag the test for update at the same time.
- **`benchmark/config.py llama_embed_url` prefers `ATLAS_VLLM_EMBED_NODEPORT`**: stage 44 fixed the gen-side resolver (`llama_url`) to prefer the new `ATLAS_VLLM_GEN_NODEPORT` over the legacy `ATLAS_LLAMA_NODEPORT` alias. The embed equivalent was missed — `llama_embed_url` only read the never-shipped `ATLAS_LLAMA_EMBED_NODEPORT`, so a fresh `atlas.conf` (which sets `ATLAS_VLLM_EMBED_NODEPORT=32736`) had its embed NodePort silently dropped, and every consumer fell back to `http://localhost:8001` regardless of the user's config. Now mirrors the gen-side order: `ATLAS_VLLM_EMBED_NODEPORT` → `ATLAS_LLAMA_EMBED_NODEPORT` (legacy) → 8001. New wire test exercises all three resolution paths.
- **CI shell-validate matches `make lint`**: the GitHub Actions workflow's `bash -n` step listed only 5 scripts; the `lint` Makefile target listed 9 (and even those didn't include all the relevant ones). A syntax bug in (e.g.) `scripts/install.sh` would surface only at deploy time, not at PR review. Both lists now cover the same 13 scripts (every entrypoint, preflight, launcher, install/uninstall/verify, model download, manifest generator, container build, latency measure, lib/config). Pinned by an invariant test that diffs the two lists and fails the moment they drift.
- **`ATLAS_LLM_PARALLEL` defaults to 1 for vLLM**: the in-code env default was `"0"` even though the comment immediately above said `"1"` was "the new default for vLLM deploys." The CLI command (`atlas/cli/commands/bench.py`) and cloud-pod Dockerfile both forced `=1` so users hitting those paths were fine, but anyone running `python -m benchmarks.v3_runner` directly fell back to a class-level `threading.Lock()` that serialized every vLLM call — completely defeating PagedAttention's concurrent-slot scheduling. Now defaults to parallel; explicit `ATLAS_LLM_PARALLEL=0` still forces single-slot for back-compat with llama.cpp-shaped backends. New wire test reloads the module under three env states (unset, =0, =1) and asserts `_parallel_mode` matches each.
- **User-facing English docs scrubbed of `Qwen3.5-9B-Q6_K` / `.gguf` references**: SETUP.md still told fresh users to `wget Qwen3.5-9B-Q6_K.gguf`; API.md examples used `"model": "Qwen3.5-9B-Q6_K"` (vLLM 4xx's that name); CONFIGURATION.md, CLI.md, and ARCHITECTURE.md listed the GGUF env defaults. After SETUP, a user would have downloaded the wrong file, then chased the model-name mismatch through every API example. All five docs now point to `QuantTrio/Qwen3.5-9B-AWQ` (the directory of safetensors shards that vLLM consumes directly) and use `qwen3.5-9b` as the served-model-name. SETUP.md replaces the wget command with `make model` (which runs `huggingface-cli download` under the hood). API.md curl example also adds `chat_template_kwargs.enable_thinking=false`. Pinned by an invariant test that forbids both `Qwen3.5-9B-Q6_K` and `.gguf` in the five canonical user docs (translations and historical reports are excluded).
- **Translated docs (ja/ko/zh-CN) caught up with English vLLM cutover**: each `docs/lang/*/README.md` and `docs/lang/*/SETUP.md` had the same six-to-seven Q6_K refs as English originally did — same broken `wget` line, same `vLLM --model models/Qwen3.5-9B-Q6_K.gguf --ctx-size --n-gpu-layers --no-mmap` bare-metal launch (a llama-server invocation that's not even a vLLM CLI), same `ATLAS_MODEL_NAME=Qwen3.5-9B-Q6_K`. Bare-metal blocks now show two `vllm serve` invocations (gen on 8000 with `--reasoning-parser qwen3 --enable-prefix-caching`, embed on 8001 with `--runner pooling --convert embed`) and use `LLAMA_GEN_MODEL=qwen3.5-9b`. Invariant test extended to the 6 translated files alongside the 5 English ones — 11 doc files total now pinned.
- **Translated TROUBLESHOOTING (ja/ko/zh-CN) cutover**: each docs/lang/*/TROUBLESHOOTING.md still asked users to `ls -la models/Qwen3.5-9B-Q6_K.gguf`, claimed the model needs ~8.2 GB VRAM (Q6_K's footprint, not the AWQ-Q4 + dual-instance reality), and showed a `response_format` curl example with `"model": "Qwen3.5-9B-Q6_K"`. All three updated: model checks now `ls -la models/Qwen3.5-9B-AWQ/`, VRAM math reflects the dual-instance reality (~12 GB gen + ~3 GB embed sharing one GPU), and the curl example uses `qwen3.5-9b` plus `chat_template_kwargs.enable_thinking=false`. Invariant test extended to cover these 3 files — 14 doc files total now pinned against `Qwen3.5-9B-Q6_K` and `.gguf` references.
- **`.env.example` documents `ATLAS_HF_CACHE` + `ATLAS_LENS_MODELS`**: both bind-mount knobs were referenced by `docker-compose.yml` but absent from `.env.example`, so a user copying `.env.example` to `.env` saw neither. Without `ATLAS_HF_CACHE` documentation, a user who didn't read the YAML would not know the ~12 GiB HuggingFace download cache lives at `./.cache/huggingface` — `docker compose down -v` or `rm -rf .cache` would silently drop it, forcing a full re-download every container restart (minutes on a fast link, hours on a slow one). New invariant test diffs the set of `${ATLAS_*}` references in compose against `.env.example` and fails on any user-tunable knob that isn't documented (with a small allow-list for inter-container service URLs that compose hardcodes).
- **Fail-fast on permanent 4xx errors from vLLM**: `benchmark/runner.py` and `benchmark/v3_runner.py` retried *every* HTTPError the same way — N attempts with exponential backoff (10s, 20s, 40s, 80s of sleep). But vLLM 4xx errors (prompt + max_tokens > max-model-len, served-model-name mismatch, validation failure on `chat_template_kwargs`) are permanent: the request body never changes between retries, so each retry is doomed to fail identically. With benchmark sweeps running 5+ tasks in parallel, a single misconfigured prompt would burn 1-3 minutes of wall-clock plus all that GPU concurrency on doomed requests before surfacing the error. Now both adapters classify status codes — `408/425/429/500/502/503/504` are retryable (genuinely transient), everything else raises `LLMConnectionError` immediately with a message naming the likely cause. New wire test (`test_v3_adapter_fails_fast_on_400_prompt_too_long`) drives an always-400 mock and asserts (a) only one HTTP attempt is made and (b) the call returns in <1 second.
- **`v3-service` LLMAdapter fail-fast on 4xx**: same fix as the benchmark runners. The V3 pipeline service had retried every HTTPError 5 times with `time.sleep(2 * (attempt + 1))` (~20s of backoff sleep on top of the doomed roundtrips). Across a V3 task that can make 50+ LLM calls, a single misconfigured prompt body would freeze the whole pipeline for 15+ minutes. Status-code classification now matches the runners — only `408/425/429/5xx` retry; all other HTTPError values propagate immediately. New wire test (`test_v3_service_adapter_fails_fast_on_400`) confirms the fix.
- **`docs/TROUBLESHOOTING.md` Performance section is no longer llama.cpp-shaped**: the "Slow Generation" troubleshooting steered users to debug `--n-gpu-layers 99` (a llama.cpp flag with no vLLM equivalent — vLLM auto-offloads every layer once `--gpu-memory-utilization` reserves a workable slice) and quoted "~51 tok/s with grammar enforcement" as the expected throughput (a llama.cpp number; vLLM uses `guided_choice`/`response_format`, not grammars, and AWQ-Q4 throughput varies with concurrency, prompt length, and DeltaNet kernel JIT-compile state — there's no honest single number to quote). The section now lists vLLM-shaped checks: CUDA visibility via `nvidia-smi`, the `Falling back to CPU` log line, concurrency knobs (`ATLAS_LLM_PARALLEL`, `ATLAS_GEN_MAX_NUM_SEQS`), and KV-cache paging symptoms. Pinned by an invariant test that forbids `--n-gpu-layers` outside its explanatory disclaimer.
- **SETUP.md K3s section reflects shipped state**: the existing "Method 3: K3s" was the V3.0 llama.cpp + spec-decode deployment recipe — `cp atlas.conf.example atlas.conf && sudo scripts/install.sh`, then `scripts/build-containers.sh && scripts/generate-manifests.sh && kubectl apply -n atlas -f manifests/`. After the V3.0.1 vLLM cutover, neither `manifests/` nor `templates/` ships in the repo, so anyone following the recipe would hit "no such file" on the first kubectl apply. The comparison table compounded the misdirection — Flash attention / q8_0+q4_0 KV / mlock / `--embeddings` are llama.cpp-only knobs with no vLLM counterpart. Section rewritten in all four languages as an honest "currently unsupported" disclaimer that points users to (a) Docker Compose on each node, (b) hand-rolled K8s manifests from `docker-compose.yml`, or (c) "wait for the K3s port — PRs welcome." New invariant test forbids the broken `kubectl apply -n atlas -f manifests/` and `scripts/generate-manifests.sh` recipes inside *fenced code blocks* in any of the four SETUP files (the disclaimer prose is allowed to reference them historically).
- **Drop fictional `atlas-launcher` script from SETUP**: all four SETUP docs ended their bare-metal section with "Start with the Launcher Script: `cp /path/to/atlas-launcher ~/.local/bin/atlas`" — but no `atlas-launcher` script exists anywhere in the repo. The `atlas` command is actually the `atlas.cli.repl:run` Python entry point that step 1 of the build (`pip install -e .`) already installs, declared in `pyproject.toml`. Section rewritten to point users at the entry point they already have. Pinned by an invariant test that forbids `atlas-launcher` in any of the four SETUP files.
- **Lens `ServerConfig.port` default aligned to 31144**: `geometric-lens/config.py` defaulted `ServerConfig.port` to `8099`, but every other surface in the stack (Dockerfile `EXPOSE 31144`, docker-compose `ATLAS_LENS_PORT=31144`, cloud-pod entrypoint `LENS_PORT=31144`, atlas-proxy hardcoded `http://localhost:31144`, every preflight curl) targets 31144. The Dockerfile and compose `command:` invoke uvicorn with explicit `--port 31144` so the container path was fine, but the `if __name__ == "__main__"` branch in `main.py` reads `config.server.port` directly — meaning the `python main.py` debug entry path silently bound to 8099, where nothing else in the stack could reach it. Default is now `int(os.environ.get("LENS_PORT", "31144"))`. Pinned by an invariant test that forbids `= 8099` as a literal default in `geometric-lens/config.py`.
- Memory note saved: `memory/project_vllm_migration.md`.

### Documentation
- Added multilingual documentation: Simplified Chinese (zh-CN), Japanese (ja), Korean (ko) for README, SETUP, and TROUBLESHOOTING
- Added language selector badges to README
- Added star history chart to Latest News section
- Rewrote README contributing section to encourage issue reports and community feedback
- Fixed V3_1_STATUS.md false claims about speed optimizations that were never applied to code

### Code Accuracy Audit
- Audited and corrected comments across 72 files for V3.0.1 accuracy
- Updated model references: Qwen3-14B to Qwen3.5-9B, embedding dimensions 5120 to 4096
- Renamed service references: rag-api to geometric-lens, Fox to llama-server
- Corrected G(x) XGBoost status: deployed and active (was incorrectly described as removed)
- Fixed normalization comments from "Fox 9B" to "Qwen3.5-9B C(x)"
- Marked legacy Fox code paths as unused in benchmark runner and geo_learning

### Test Fixes
- Fixed embedding dimensions in test fixtures (5120 to 4096)
- Fixed geometric-lens port in test conftest (8001 to 8099)
- Updated DivSampling test assertions to match actual 4+4+4 perturbation counts
- Corrected G(x) cost field parameter count: ~2.16M / 8.3MB (was ~2.7M / 10MB)

## [3.0.1] - 2026-04-05

### Tool-Call Agent Loop Architecture
- Replaced Aider format-translation proxy with structured JSON tool-call agent loop
- Grammar-constrained output via llama-server `response_format:json_object` — 100% valid JSON
- 8 tool definitions: `read_file`, `write_file`, `edit_file`, `delete_file`, `run_command`, `search_files`, `list_directory`, `plan_tasks`
- Per-file tier classification: T1 (config/data) writes directly, T2 (logic/features) routes through V3 pipeline
- 3400+ lines new Go code across 12 files in `atlas-proxy/`

### V3 Pipeline Integration
- All 14 V3 steps wired into `write_file`/`edit_file` executors for T2/T3 files
- PlanSearch → DivSampling → Budget Forcing → Build Verification → C(x)/G(x) Scoring → Best-of-K → S*/Blend-ASC → Failure Analysis → PR-CoT Repair → Refinement Loop → Derivation Chains → Metacognitive → Final Write
- Per-file-type build verification: tsc, py_compile, gcc, go build, cargo check, bash -n
- V3 service SSE streaming: pipeline progress visible in real-time

### CLI Experience
- `atlas` command: starts all services and launches Aider
- Streaming progress: `[Turn N/M]` with tool call details, V3 pipeline steps, completion summary
- Exploration budget: 4 consecutive read-only calls triggers nudge, prevents model from over-exploring
- Pre-injected project context: model sees project file list in system prompt
- File deletion via fast-path before tier classification
- Truncation prevention: 32K context, reject write_file for existing files >100 lines, detect truncated args before execution

### Deployment
- Docker Compose (`docker-compose.yml`) for full stack orchestration
- Podman compatible with host networking
- `.env.example` with all configurable parameters
- `atlas` script auto-detects Docker vs bare-metal and routes accordingly

### Renames (362 total reference updates)
- `rag-api/` → `geometric-lens/` (directory + all references)
- `ATLAS_RAG_URL` → `ATLAS_LENS_URL`
- `ATLAS_FOX_URL` → `ATLAS_INFERENCE_URL`
- `foxURL` → `inferenceURL` (Go code)
- `ralph-loop` → `verify-repair loop`
- `rag.py` → `pipeline.py` (geometric-lens orchestration)

### Reliability
- 8-level test × 3 iterations: 95.8% (23/24)
- 5-language integration: 100% (Shell, Python, Rust, C, Go)
- L6 (add feature to existing project): 67% — marked as future improvement

### Documentation Overhaul
- **ARCHITECTURE.md**: Complete rewrite — 13 Mermaid diagrams (service topology, agent loop flow, V3 pipeline, module map, sequence diagrams), every component verified against source code
- **API.md**: Complete rewrite — every endpoint across all 5 services verified against source, request/response formats, SSE stages
- **CLI.md**: Complete rewrite — startup flow diagram, streaming format, workflow examples, troubleshooting, env vars, Aider config reference
- **CONFIGURATION.md**: Complete rewrite — every env var across all services verified, internal constants, Docker Compose vs K3s differences
- **MAP.md**: Complete rewrite — every file in repo with clickable tree, 150 file links, 18 description tables
- **SETUP.md**: Complete rewrite — verified build steps, first-run guide, bare metal, K3s, hardware sizing, Lens training guide
- **TROUBLESHOOTING.md**: Complete rewrite — quick diagnostics, 20+ issue scenarios with verified fixes
- **README.md**: Honest 7-step setup with actual download command, prerequisites, model clarity (Qwen3-14B vs Qwen3.5-9B)
- Reorganized historical docs into `docs/reports/` (ablation studies, status tracking, migration guides)

### Bug Fixes
- **geometric-lens Dockerfile port mismatch**: Container was listening on 8001 but docker-compose expected 8099 — fresh Docker Compose deploys had broken Lens service. Fixed Dockerfile to use port 8099.
- **Python CLI default RAG port**: `atlas/cli/client.py` defaulted to port 31144 (K3s NodePort) instead of 8099 (Docker Compose). Fixed default to match Docker Compose.
- **Missing Aider config files**: `.aider.model.settings.yml` and `.aider.model.metadata.json` were not in the repo — the `atlas` launcher would fail without them. Restored both files and added `.gitignore` exceptions.
- GitHub Issue #6: `hostname -I` → portable fallback chain (`ip addr` → `hostname -I` → `hostname -i`) for Arch Linux compatibility
- GitHub Issue #10: `rag-api/` → `geometric-lens/` restructuring resolved missing models directory
- GitHub Issue #11: Added Geometric Lens training documentation to SETUP.md with HuggingFace dataset link
- GitHub Issue #12 / PR #13: `docker image exists` → `docker image inspect` in build script

### Cleanup
- Removed 62 stale test directories, old v1 proxy binary, dead G(x) metric tensor training scripts
- Removed stale tests for deleted services (api-portal, dashboard, embedding-service, task-worker)
- Removed root-level development artifacts (bubble_sort.py, snake_game.py, etc.)
- All hardcoded `/home/isaac/` paths replaced with `$HOME` or `ATLAS_DIR` env vars

## [3.0] - 2026-03-05

### V3.0 Benchmark Release
- **74.6% LCB pass@1** (447/599) on frozen Qwen3-14B
- Full ablation study: conditions A–D with per-task results
- Phase 1 (PlanSearch/DivSampling): +12.4pp
- Phase 3 (PR-CoT/Refinement/Derivation): +7.3pp
- Self-verified Phase 3 using model-generated test cases

## [2.5.1] - 2026-02-23

### Confirmation Ablation: Embedding Source Hypothesis — STRONG CONFIRMATION
- **H1: Self-embeddings restore C(x) discrimination: CONFIRMED (+39.5pp)**
  - C(x) selects passing candidate 87.8% on mixed-result tasks vs 48.3% random (p < 0.000001)
  - V2.5 result (+0.6pp under nomic 768-dim) was an embedding source limitation, not architecture failure
  - Reverse energy selects only 4.3%, proving strong directional signal
  - Val AUC: 0.9934, energy separation: 21.75 (7.2x wider than V2.5)
- **H2: G(x) adds value beyond C(x): NEUTRAL (0.0pp)**
  - G(x) contributes zero at optimal alpha (0.001); monotonically degrades at higher alpha
  - Zero corrections, zero breakages across all mixed-result tasks
- **Outcome B**: Ship C(x)-only with self-embeddings, remove or redesign G(x)
- **Difficulty routing validated**: Q1 (low energy) = 100% oracle, Q4 (high energy) = 0.3%
- **C(x) confirmed as both verifier (87.8% selection) and router (perfect difficulty stratification)**
- Runtime: 24h 42m on LiveCodeBench v5 (599 tasks, K=3, 4 epochs)
- Infrastructure: Qwen3-14B with `--embeddings` (no spec decode, ~45 tok/s)
- Risk R6 (Lens non-discriminating) RESOLVED; Risk R11 (no verifier) substantially mitigated

## [2.5.0] - 2026-02-21

### Ablation Study
- Systematic ablation of Geometric Lens, router, and infrastructure components
- Finding: C(x) energy scoring ≈ random for candidate selection under nomic embeddings (37.7% vs 37.1%, within 3.4pp seed variance) — **V2.5.1 confirmed this was an embedding source limitation** (87.8% accuracy restored with self-embeddings)
- Finding: C(x) energy strongly correlates with task difficulty (58.5% vs 18.9% pass rate across tiers)
- Finding: G(x) metric tensor confirmed dormant (5.2M params, zero impact)
- Finding: Pattern cache bypassed entirely by benchmark runner

### Architecture Change
- Discovered `--embeddings` flag breaks speculative decoding (forces n_batch=512)
- Migrated to two-server sidecar architecture: generation + spec decode on Server A, embeddings via nomic-embed-text-v1.5 on Server B
- Recovered ~2.6x generation throughput (~38 tok/s → ~100 tok/s)
- Net VRAM delta: approximately -230 MiB (sidecar cheaper than --embeddings overhead)

## [2.0.0] - 2026-02-18

### Architecture Changes
- Replaced Qdrant vector DB + embedding service with PageIndex tree-based RAG
- Added Geometric Lens (Cost Field + Metric Tensor) for candidate quality prediction
- Added Confidence Router with difficulty-based adaptive-k selection
- Added Pattern Cache (Redis + Ebbinghaus memory decay)
- Added Best-of-K pipeline with parallel candidate generation
- Added sandboxed code execution for benchmark evaluation
- Added speculative decoding with Qwen3-0.6B draft model
- Added KV cache quantization (q4_0)

### Benchmark Results (Run ID: v2_run_20260217_125310)
- LiveCodeBench: 36-41% pass@1 (across Lens training epochs, k=3)
- GPQA Diamond: 47.0% (k=5)
- SciCode: 14.7% sub-problems (341 tasks, k=1)
- Geometric Lens: 0.968 Val AUC, ~80% first-pick accuracy (151/188)
- Throughput: 109 tasks/hr on RTX 5060 Ti 16GB

### Removed
- Qdrant vector database
- MiniLM-L6-v2 embedding service
- LoRA nightly training pipeline (moved to v1_archived/, CronJob suspended)
- V1 benchmark suite (HumanEval, MBPP, Custom)

### Fixed Post-Release
- mlock allocation failure — added LimitMEMLOCK=infinity systemd override for K3s
- Speculative decode slot 1 failure — quantized draft KV cache to q4_0 (-ctkd/-ctvd)
- Dashboard crash-loop — fixed missing Jinja2 default filters

### Notes
- IFBench evaluation incomplete (excluded from results)
- All results from single benchmark run (variance unknown)

## [1.0.0] - 2026-02-04

Initial release. See benchmark/v1_benchmark_report.md for V1 results.
