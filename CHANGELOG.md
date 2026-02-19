# Changelog

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
- LiveCodeBench: 36-41% pass@1 (across Lens training epochs)
- GPQA Diamond: 47.0%
- SciCode: 14.7% sub-problems, 5.0% main problems
- Geometric Lens: 0.968 Val AUC, 100% selection efficiency (188/188)
- Throughput: 109 tasks/hr on RTX 5060 Ti 16GB

### Removed
- Qdrant vector database
- MiniLM-L6-v2 embedding service
- LoRA nightly training pipeline
- V1 benchmark suite (HumanEval, MBPP, Custom)

### Known Issues
- IFBench evaluation incomplete (excluded from results)
- All results from single benchmark run (variance unknown)
- mlock allocation fails on current kernel config
- Speculative decode slot 1 failure (non-blocking)

## [1.0.0] - 2026-02-04

Initial release. See benchmark/v1_benchmark_report.md for V1 results.
