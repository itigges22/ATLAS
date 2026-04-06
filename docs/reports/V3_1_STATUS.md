# ATLAS V3.1 Implementation Status

**Last Updated**: 2026-03-07 (morning)
**Current Phase**: READY FOR FULL BENCHMARK — PlanSearch + ReASC fixed for 9B, all ablation conditions verified
**Goal**: Publication-ready ablation on Qwen3.5-9B, target 80-90% LCB pass@1
**Tests**: 854/854 V3 tests passing

## Phase 0: Pre-Ablation Fixes (On 14B)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 1.1 | Retrain Lens on real LCB data | COMPLETE | 4096-dim model: Val AUC 0.7229 (321 PASS + 50 synthetic FAIL). Deployed in geometric-lens. Will retrain on real 9B FAIL data after 50-task run. |
| 1.2 | Fix SandboxAdapter stdio for S* | COMPLETE | SStarSandboxAdapter pipes specific stdin for stdio tasks, compares stdout |
| 1.3 | Remove G(x) metric tensor | COMPLETE | Deleted metric_tensor.py, correction.py. Cleaned service.py, training.py, main.py. Backward-compat aliases kept. |
| 1.4 | Candidate selection baselines | COMPLETE | 4 strategies (lens/random/logprob/oracle), 17/17 tests pass, --selection-strategy CLI arg |
| 1.5 | Per-task latency instrumentation | COMPLETE | Timing for probe, phase2_alloc, phase1_gen, sandbox, self_test_gen, pr_cot, refinement, derivation, phase3_total |

## Phase 1: 9B Model Swap Infrastructure

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 2.1 | Model selection + quantization | COMPLETE | Qwen3.5-9B-Q6_K.gguf (7.0 GB) from unsloth/Qwen3.5-9B-GGUF |
| 2.2 | Validate patched llama.cpp | COMPLETE | Container v3.1-9b built with latest llama.cpp (DeltaNet SSM support), deployed to K3s |
| 2.3 | Benchmark raw throughput | COMPLETE | 48.3 tok/s warm (no spec decode), VRAM: 10819/16311 MiB (5.4 GB free) |
| 2.4 | Verify prompt formatting | COMPLETE | ChatML works, /nothink still produces empty think tags (stripped by BudgetForcing), embeddings=4096-dim |
| 2.5 | Update config + rebuild | COMPLETE | entrypoint-v3.1-9b.sh, Dockerfile.v31, deploy-9b.sh, smoke-test-9b.sh, ConfigMap updated |

## Speed Optimizations

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 4.1 | Parallel candidate generation | COMPLETE | ThreadPoolExecutor(max_workers=3) for DivSampling fill, thread-safe LLMAdapter per thread |
| 4.2 | Pipeline Lens + sandbox | COMPLETE | Sandbox tests + embedding storage run in parallel ThreadPoolExecutor(max_workers=3), interleaved with scoring. Deployed and verified in 10-task smoke test. |
| 4.3 | Phase 3 early exit | COMPLETE | Already implemented — PR-CoT/refinement success returns before subsequent stages |
| 4.4 | Self-test caching | COMPLETE | Already implemented — self-tests generated once, reused across PR-CoT and refinement |
| 4.5 | Verification (50-task) | COMPLETE | 50-task: 33/50 (66.0%), 32 phase1, 1 pr_cot, 4 error (OOM restarts). 10-task smoke: 8/10 (80.0%). Server stable with mlock + 14Gi + 16K ctx. |
| 4.6 | Sandbox-based probe early exit | COMPLETE | Probe sandbox-tested BEFORE k allocation. Passes → k=1 (skip PlanSearch), fails → k=3 (PlanSearch runs). Replaces broken energy-based ReASC gate. |
| 4.7 | PlanSearch timeout fix | COMPLETE | Dedicated LLMAdapter(timeout=300) for PlanSearch calls. Fixed _get_max_tokens bug (steps 1/2: 4096→2048). PlanSearch generates 9-13K tokens on hard tasks. |
| 4.8 | BlendASC bypass for failed probe | COMPLETE | When probe fails sandbox, k=3 regardless of BlendASC (energy-based k allocation uninformative on 9B). BlendASC still logged for telemetry. |
| 4.9 | PlanSearch step parallelization | COMPLETE | Steps 2+3 run in parallel via ThreadPoolExecutor. Code in plan_search.py. |
| 4.10 | Phase 3 parallel strategies | COMPLETE | PR-CoT, refinement, derivation run simultaneously with separate LLMAdapter instances. First winner takes all. |
| 4.11 | DeltaNet multi-slot serialization | COMPLETE | Class-level threading.Lock() in LLMAdapter._send_request() serializes /completion calls. DeltaNet hangs when 2+ slots generate simultaneously (llama.cpp bug). Lock ensures full 48 tok/s single-slot throughput. |
| 4.12 | Probe MAX_TOKENS reduction | COMPLETE | Reduced from 8192 to 4096. At 24 tok/s long-gen, 180s timeout only allows ~4300 tokens — 8192 caused consistent timeouts on hard tasks. |

## Telemetry Improvements

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 5.1 | Unified telemetry schema | COMPLETE | v3_events.jsonl merges all telemetry sub-fields, per-candidate energies, selection strategy, latency breakdown |
| 5.2 | Raw embedding storage | COMPLETE | Binary .emb format (struct, no numpy), 4/4 tests pass, wired into pipeline |
| 5.3 | PlanSearch constraint logs | COMPLETE | Constraint sets logged to telemetry per task (plansearch_constraints field) |

## Lens Evolution Wiring (Phase 4)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 6.1 | Wire replay buffer + EWC | COMPLETE | --enable-feedback CLI flag, LensFeedbackCollector fully wired: record->retrain->propagate |
| 6.2 | Validation run | NOT STARTED | 599-task with vs without |
| 6.3 | Risk assessment | NOT STARTED | EWC lambda tuning |

## Analysis Pipeline (Section 8)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 8.1 | Ablation analysis module | COMPLETE | bootstrap_delta, ablation_table, pairwise_significance, phase_waterfall, latency_summary, full_report |
| 8.2 | Bootstrap significance testing | COMPLETE | Paired bootstrap (shared task_ids), unpaired fallback, 10K resamples, two-sided p-value |
| 8.3 | Automated figure generation | NOT STARTED | Needs matplotlib or similar; defer until data exists |
| 8.4 | Cost analysis | EXISTING | benchmark/analysis/cost_analysis.py already has CostAnalyzer |

## Documentation & Config Updates

| Item | Status | Notes |
|------|--------|-------|
| atlas.conf.example: ATLAS_V3_SELECTION_STRATEGY | COMPLETE | "lens" default, 4 strategies |
| atlas.conf.example: ATLAS_V3_ENABLE_FEEDBACK | COMPLETE | false default, Phase 4 toggle |
| docs/CONFIGURATION.md: Benchmark Runner section | COMPLETE | Selection strategy + feedback documented |
| docs/ARCHITECTURE.md: G(x) weight path fix | COMPLETE | Removed stale metric_tensor.pt reference |
| docs/ARCHITECTURE.md: S* stdio fix note | COMPLETE | Updated Phase 2 description with SStarSandboxAdapter |
| conftest.py: conditional infra imports | COMPLETE | V3 tests run without redis/httpx |

## Errors & Fixes Log

| Date | Error | Fix | Impact |
|------|-------|-----|--------|
| 2026-03-05 | huggingface-cli not in PATH | Use Python API hf_hub_download() directly | Model download works |
| 2026-03-05 | kubectl TLS handshake timeout | Use KUBECONFIG=/etc/rancher/k3s/k3s.yaml | K3s access reliable |
| 2026-03-05 | Duplicate podman builds competing for CPU | Killed stale process (PID 2973313) | Build speed restored |
| 2026-03-05 | Failed tasks have no stored code | Synthetic FAIL samples via truncation in retrain script | Lens retrain possible |
| 2026-03-06 | geometric-lens had 103 restarts (old image) | Rebuilt with 4096-dim cost_field.pt, redeployed | 0 restarts, Lens serving 4096-dim |
| 2026-03-06 | LLM timeout (2 of 10 tasks in smoke) | 600s timeout × 4 retries = 2400s worst-case. Normal for hard tasks at 48 tok/s. | Not blocking; monitor in 50-task run |
| 2026-03-06 | --smoke ignores --max-tasks | --smoke always takes first 10 tasks; use --max-tasks alone for custom counts | User awareness |
| 2026-03-06 | LLM timeout too generous for 9B | Reduced from 600s×4 retries to 180s×2 retries (+3 extra for 503). Caps stuck-task at ~6 min. | Full benchmark efficiency |
| 2026-03-06 | Universal think pre-fill killed Phase 3 | Removed from LLMAdapter; only BudgetForcing.format_chatml() handles nothink pre-fill. Phase 3 needs thinking. | 80% pass rate restored (was 20-30%) |
| 2026-03-06 | 503 Service Unavailable (slots full) | Added 503-specific retry with short backoff in LLMAdapter._send_request(). Reduced DivSampling workers from 3 to 2. | No more 503 errors |
| 2026-03-06 | MemoryPressure TaintManagerEviction | Removed --mlock/--no-mmap from entrypoint, reduced context 32K→16K, memory limit 16Gi→12Gi. RAM: 4.6→3.6 GB. | Stable server, no evictions |
| 2026-03-06 | ConfigMap shebang mangled | Shell expansion escaped ! in #!/bin/bash. Fix: write to file first, then --from-file. | Pod starts correctly |
| 2026-03-06 | OOMKilled at 12Gi limit | Without mlock, process RSS grows unbounded during inference. Fix: mlock + 14Gi limit + 16K context. | Stable server, no OOM |
| 2026-03-06 | Lens retrain poor on 46 samples | 46-sample 9B-only dataset: Val AUC 0.33. Fix: combined 597 samples (46 real 9B + 471 14B re-embedded + 80 synthetic FAIL). Val AUC 0.7417. | Better Lens discrimination |
| 2026-03-06 | Connection refused during fast sequential runs | Baseline tasks completing instantly hit llama-server before it accepted connections. Fix: improved retry logic for ConnectionError/OSError with progressive backoff. | Resilient to transient failures |
| 2026-03-06 | Lens pre-flight UNAVAILABLE on first check | score-text endpoint timed out during lazy model load. Fix: added retry with 3s delay in pre-flight check. | Reliable pre-flight |
| 2026-03-06 | Baseline token count always 0 | `llm.total_tokens` overrode accumulated `result["total_tokens"]` from fallback path. Fix: `max(result["total_tokens"], llm.total_tokens)`. | Correct telemetry |
| 2026-03-06 | llama-server pod evicted (NodeNotReady) | Node memory pressure caused pod eviction. New pod stuck Pending: nvidia.com/gpu not in node capacity. Fix: restart NVIDIA device plugin DaemonSet pod. | GPU re-registered, server recovered |
| 2026-03-07 | ReASC always sets k=1 on 9B | Probe energy ~1-4 raw normalizes to ~0.01-0.04, always below 0.10 threshold. Fix: sandbox-based probe early exit — test probe in sandbox BEFORE k allocation. If passes → k=1, if fails → k=3. | PlanSearch now runs on hard tasks |
| 2026-03-07 | BlendASC sets k=1 on 9B | Same normalization issue as ReASC — all 9B probe energies fall in [0,0.10) "easy" range. Fix: when probe fails sandbox, always use k=3 regardless of BlendASC output. Log blend_asc_k for telemetry. | 3 candidates generated for hard tasks |
| 2026-03-07 | PlanSearch timeout on 9B | 180s HTTP timeout insufficient for long competition prompts at 47 tok/s. Fix: PlanSearch uses dedicated LLMAdapter with 300s timeout. Also fixed _get_max_tokens bug (steps 1/2 used 4096 instead of intended 2048). | PlanSearch generates code on 9B (9581-13090 tokens) |
| 2026-03-07 | LCB/1883_C newly solvable | Was FAIL in 50-task run (k=1 due to ReASC). Now PASS via PlanSearch candidate #2 with k=3. | +1 task solved by PlanSearch diversity |
| 2026-03-07 | DeltaNet multi-slot generation hang | llama-server hangs when 2+ slots generate simultaneously (cont-batching). Server log: `n_tokens = -16` in ctx checkpoint. Tried `--ctx-checkpoints 0` (no fix), `--no-cont-batching` (worked but 14.9 tok/s). Fix: class-level `threading.Lock()` in `LLMAdapter._send_request()` serializes all /completion calls. Full single-slot throughput maintained. | Stable 4-slot server with cont-batching |
| 2026-03-07 | Long-generation throughput degradation | Short gens (~500 tok) run at 48 tok/s, long gens (4000+ tok) degrade to ~22-24 tok/s. DeltaNet hybrid has attention layers with O(n) per-token compute as sequence grows. Not a bug — architecture-inherent. | Hard tasks take ~12-14 min (vs 10 min target) |
| 2026-03-07 | Probe MAX_TOKENS too high | MAX_TOKENS=8192 caused probe timeouts: at 24 tok/s long-gen, 180s timeout only allows ~4300 tokens. Fix: reduced MAX_TOKENS from 8192 to 4096 (most responses <1000 tok). | Prevents probe timeout on long prompts |

## Ablation Design (9B)

| Condition | Configuration | Status | Result |
|-----------|--------------|--------|--------|
| A | Bare Qwen3.5-9B, k=1 | SMOKE VERIFIED | 3/3 (smoke) |
| B | + Phase 1 (PS+DS+BF), k=3 | SMOKE VERIFIED | 3/3 (smoke) |
| C | + Retrained Lens (on B data) | SMOKE VERIFIED | 3/3 (smoke) |
| D | + Phase 2 (BASC+ReASC+S*) | SMOKE VERIFIED | 3/3 (smoke) |
| E | + Phase 3 (PR-CoT+refinement) | SMOKE VERIFIED | 3/3 (smoke) |
| F | Full pipeline | SMOKE VERIFIED | 3/3 (smoke) |

## Go/No-Go Gates

| Gate | Trigger | Pass Criterion | Result |
|------|---------|---------------|--------|
| 9B Baseline | Condition A | Within 10pp of 14B (54.9%) | -- |
| Phase 1 Value | Condition B | +5pp over A (significant) | -- |
| Lens Value | Condition C | Lens > random by +2pp | -- |
| Phase 2 Verdict | Condition D | Any positive delta | -- |
| Phase 3 Value | Condition E | +3pp over best of B/C/D | -- |
| Composition | Condition F | F >= max(D, E) | -- |

## Infrastructure State

- **Model**: Qwen3.5-9B-Q6_K.gguf deployed at /home/isaac/models/ (7.0 GB)
- **Container**: localhost/llama-server:v3.1-9b DEPLOYED (Dockerfile.v31 + latest llama.cpp)
- **geometric-lens**: Rebuilt with Phase 0 cost_field.pt (Val AUC 0.9467), Lens serving correctly
- **GPU**: RTX 5060 Ti 16GB, ~10.7/16.3 GB used
- **Cluster**: llama-server (0 restarts, 8h+ uptime), geometric-lens, redis, sandbox all Running
- **Embedding dim change**: 5120 (14B) -> 4096 (9B), Lens C(x) auto-adapts via input_dim param
- **No spec decode**: Qwen3.5 DeltaNet architecture not supported by llama.cpp spec decode yet
- **Throughput**: 48 tok/s warm (short gen), 22-24 tok/s (long gen 4000+ tok), --parallel 4, context 163840 (40K/slot), mlock enabled
- **DeltaNet serialization**: Class-level threading.Lock() in LLMAdapter — prevents multi-slot generation hang. 4 slots for connection acceptance, 1 slot active at a time.
- **10-task verification**: 10/10 (100.0%) pass@1 — 9 phase1, 1 pr_cot, zero errors
- **Lens 4096-dim**: Phase 0 retrain Val AUC 0.9467, 597 samples (504 PASS + 93 FAIL), contrastive ranking loss, pass energy 0.59 / fail energy 1.20
- **Memory config**: mlock + 14Gi limit + 40K/slot context. Prevents both OOM and eviction.
- **MAX_TOKENS**: 4096 (probe/baseline). At 24 tok/s long-gen, 180s timeout = ~4300 tok max. Prevents probe timeouts.
- **50-task verification**: 33/50 (66.0%) pass@1, 14 tasks/hr, 4 errors (OOM-related, fixed)
- **Tests**: 854/854 V3 tests passing
- **Ablation conditions A-F**: All 6 smoke-verified (3/3 each), re-verified after PlanSearch fix
- **Crash recovery**: Tested — resume from per-task JSON files works correctly
- **Retry logic**: 503 + ConnectionError + OSError with progressive backoff (5 attempts max)
- **Probe early exit**: Sandbox-tests probe before k allocation. Easy tasks (probe passes) → k=1, ~10s. Hard tasks → k=3, PlanSearch runs.
- **PlanSearch on 9B**: Working. Generates 2 constraint sets + 3 candidates. 9-13K tokens per hard task. 300s timeout.
- **Full pipeline 5-task**: 5/5 (100%), 3 phase1 + 1 refinement + 1 PlanSearch-contributed
- **Full pipeline 10-task (verify_maxtok)**: 10/10 (100.0%), 9 phase1 + 1 pr_cot. Zero errors. Both prior timeouts (LCB/1883_C, LCB/1899_B) now pass.
- **Speed**: Easy tasks ~10-25s (probe with "light" thinking), hard tasks ~5-11 min (2.5x faster than before)
- **Speed optimizations (2026-03-15)**: Probe "nothink"→"light" tier, Phase 3 budgets reduced (refinement 5→2 iters/120s, PR-CoT 3→2 rounds, derivation 5→3 subproblems), EmbedAdapter retry logic (3x with backoff), Phase 0 C(x) deployed (AUC 0.9467)
- **Phase 3 empirical (9B)**: 0% success rate on 60-task gx-phase1 run (PR-CoT 0/15, refinement 0/15, derivation 0/16). Budgets reduced to minimize waste without removing capability.
- **Embedding storage fixed**: EmbedAdapter retry logic produces 3.0 embeddings/task (was 0.3/task before)
- **Disk**: ~20 GB free on /home
