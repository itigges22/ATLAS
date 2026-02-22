# Changelog

## [2.5.1] - Planned

### Investigation: Embedding Source Hypothesis
- **Blocking dependency for V3 Phase 4 strategy**
- V2.5 ablation found C(x) ≈ random for candidate selection under 768-dim nomic embeddings
- Hypothesis: discrimination was lost when embedding source switched from Qwen3-14B self-embeddings (5120-dim) to nomic-embed-text-v1.5 (768-dim) in V2.5 — NOT a fundamental Lens failure
- Confirmation ablation planned: re-run V2.5 methodology with original self-embeddings
- If confirmed: restore self-embeddings without breaking spec decode (hidden state extraction, post-generation embedding, draft model embeddings, or hybrid approach)
- Success criteria: selection accuracy > random + 5pp; spec decode ≥80 tok/s; VRAM ≤16 GiB

## [2.5.0] - 2026-02-21

### Ablation Study
- Systematic ablation of Geometric Lens, router, and infrastructure components
- Finding: C(x) energy scoring ≈ random for candidate selection under nomic embeddings (37.7% vs 37.1%, within 3.4pp seed variance) — under investigation by V2.5.1
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
