# ATLAS V1 Benchmark Report

**Generated:** 2026-02-05 02:16:35
**Run ID:** full_run_20260204_140715
**Total Runtime:** 12h 9m
**Status:** COMPLETE

---

## Run Health

| Metric | Value |
|--------|-------|
| Total Crashes | 0 |
| Phases with Retries | None |
| Phases Failed (3 strikes) | None |
| Total Wall-Clock Time | 12h 9m |

---

## Target Metrics Summary

| Category | Metric | Target | Description |
|----------|--------|--------|-------------|
| **Accuracy** | pass@1 | ≥65% (HumanEval), ≥70% (MBPP) | First attempt success rate |
| **Accuracy** | pass@5 | ≥95% | Success within 5 attempts |
| **Accuracy** | pass@20 | ≥99.5% | Success within 20 attempts |
| **Performance** | Throughput | ≥100 tasks/hr | Tasks completed per hour |
| **Performance** | Time to Solution | <60s median | Wall-clock to verified success |
| **Cost** | Cost Efficiency | ≥30x cheaper | Cloud cost / Local cost ratio |
| **Efficiency** | Tokens/Watt-Hour | Baseline | Novel metric (establishing) |
| **Efficiency** | Tasks/Watt-Hour | Baseline | Novel metric (establishing) |

---

## Benchmarks Executed

| Benchmark | Tasks | Attempts | Runs | Status |
|-----------|-------|----------|------|--------|
| Humaneval pass@1 | 164 | 1 | 1 | ✓ |
| Mbpp pass@1 | 500 | 1 | 1 | ✓ |
| Custom pass@1 | 100 | 1 | 1 | ✓ |
| Humaneval pass@20 (run 1) | 164 | 20 | 1 | ✓ |
| Humaneval pass@20 (run 2) | 164 | 20 | 1 | ✓ |
| Humaneval pass@20 (run 3) | 164 | 20 | 1 | ✓ |
| Custom pass@20 (run 1) | 100 | 20 | 1 | ✓ |
| Custom pass@20 (run 2) | 100 | 20 | 1 | ✓ |
| Custom pass@20 (run 3) | 100 | 20 | 1 | ✓ |

---

## Pass@1 Results

### Humaneval

## Pass@k Results: humaneval

- Total Tasks: 164
- Samples per Task: 1

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 99.4% | ≥65.0% | ✓ | [98.2%, 100.0%] |
| pass@5 | 99.4% | ≥95.0% | ✓ | [19.6%, 20.0%] |
| pass@10 | 99.4% | — | — | [9.8%, 10.0%] |
| pass@20 | 99.4% | ≥99.5% | ✗ | [4.9%, 5.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 99.4% | 67.0% | +32.4% |
### Mbpp

## Pass@k Results: mbpp

- Total Tasks: 500
- Samples per Task: 1

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 55.4% | ≥70.0% | ✗ | [51.2%, 59.6%] |
| pass@5 | 55.4% | ≥95.0% | ✗ | [10.2%, 12.0%] |
| pass@10 | 55.4% | — | — | [5.1%, 6.0%] |
| pass@20 | 55.4% | ≥99.5% | ✗ | [2.6%, 3.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 55.4% | 72.0% | -16.6% |
### Custom

## Pass@k Results: custom

- Total Tasks: 100
- Samples per Task: 1

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 66.0% | ≥65.0% | ✓ | [56.0%, 76.0%] |
| pass@5 | 66.0% | ≥95.0% | ✗ | [11.4%, 15.0%] |
| pass@10 | 66.0% | — | — | [5.7%, 7.5%] |
| pass@20 | 66.0% | ≥99.5% | ✗ | [2.8%, 3.7%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 66.0% | 0.0% | +66.0% |

---

## Pass@k Results (k=20)

### Humaneval (aggregated over 3 runs)

#### Run 1

## Pass@k Results: humaneval

- Total Tasks: 164
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 99.8% | ≥65.0% | ✓ | [99.7%, 99.9%] |
| pass@5 | 100.0% | ≥95.0% | ✓ | [100.0%, 100.0%] |
| pass@10 | 100.0% | — | — | [100.0%, 100.0%] |
| pass@20 | 100.0% | ≥99.5% | ✓ | [100.0%, 100.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 99.8% | 67.0% | +32.8% |
#### Run 2

## Pass@k Results: humaneval

- Total Tasks: 164
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 99.9% | ≥65.0% | ✓ | [99.8%, 100.0%] |
| pass@5 | 100.0% | ≥95.0% | ✓ | [100.0%, 100.0%] |
| pass@10 | 100.0% | — | — | [100.0%, 100.0%] |
| pass@20 | 100.0% | ≥99.5% | ✓ | [100.0%, 100.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 99.9% | 67.0% | +32.9% |
#### Run 3

## Pass@k Results: humaneval

- Total Tasks: 164
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 99.6% | ≥65.0% | ✓ | [99.2%, 99.9%] |
| pass@5 | 100.0% | ≥95.0% | ✓ | [100.0%, 100.0%] |
| pass@10 | 100.0% | — | — | [100.0%, 100.0%] |
| pass@20 | 100.0% | ≥99.5% | ✓ | [100.0%, 100.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 99.6% | 67.0% | +32.6% |
### Custom (aggregated over 3 runs)

#### Run 1

## Pass@k Results: custom

- Total Tasks: 100
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 77.3% | ≥65.0% | ✓ | [70.0%, 83.8%] |
| pass@5 | 86.9% | ≥95.0% | ✗ | [80.2%, 92.6%] |
| pass@10 | 90.1% | — | — | [84.1%, 95.2%] |
| pass@20 | 93.0% | ≥99.5% | ✗ | [88.0%, 98.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 77.3% | 0.0% | +77.3% |
#### Run 2

## Pass@k Results: custom

- Total Tasks: 100
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 76.7% | ≥65.0% | ✓ | [68.9%, 83.4%] |
| pass@5 | 84.2% | ≥95.0% | ✗ | [76.9%, 90.5%] |
| pass@10 | 86.5% | — | — | [79.5%, 92.8%] |
| pass@20 | 88.0% | ≥99.5% | ✗ | [82.0%, 94.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 76.7% | 0.0% | +76.7% |
#### Run 3

## Pass@k Results: custom

- Total Tasks: 100
- Samples per Task: 20

| Metric | Score | Target | Status | 95% CI |
|--------|-------|--------|--------|--------|
| pass@1 | 78.2% | ≥65.0% | ✓ | [70.8%, 84.9%] |
| pass@5 | 86.6% | ≥95.0% | ✗ | [80.4%, 92.6%] |
| pass@10 | 88.6% | — | — | [82.1%, 94.2%] |
| pass@20 | 90.0% | ≥99.5% | ✗ | [84.0%, 96.0%] |

## Comparison with Baseline

| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |
|--------|----------|-------------------|------------|
| pass@1 | 78.2% | 0.0% | +78.2% |

---

## Cost Analysis

### HUMANEVAL pass@1

## Cost Analysis

### Summary

- Total Tasks: 164
- Successful Tasks: 163
- Total Tokens: 13,125
- Total Inference Time: 310.0s
- Total Wall Time: 312.1s
- Estimated Energy: 0.0124 kWh

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 1891.4 tasks/hr | ≥100 tasks/hr | ✓ |
| Time to Solution (median) | 1.6s | <60s | ✓ |

### Local Cost

- Cost per Successful Task: $0.000025

### Novel Efficiency Metrics (ATLAS Baselines)

- **Tokens per Watt-Hour:** 1,059
- **Tasks per Watt-Hour:** 13.15

*These are novel metrics establishing ATLAS V1 baselines.*

### Cloud API Cost Comparison

| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |
|----------|------------|------------|-------|---------------|
| claude-haiku | $0.0230 | $0.0041 | 5.5x | ✗ |
| claude-sonnet | $0.2756 | $0.0041 | 66.5x | ✓ |
| gpt-4o | $0.3281 | $0.0041 | 79.2x | ✓ |
| gpt-4o-mini | $0.0118 | $0.0041 | 2.9x | ✗ |

**Cost Efficiency Target: ≥30x cheaper than cloud APIs**
**Status: ✗ Target not met** (minimum ratio: 2.9x)

### MBPP pass@1

## Cost Analysis

### Summary

- Total Tasks: 500
- Successful Tasks: 277
- Total Tokens: 31,383
- Total Inference Time: 726.0s
- Total Wall Time: 735.7s
- Estimated Energy: 0.0290 kWh

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 2446.8 tasks/hr | ≥100 tasks/hr | ✓ |
| Time to Solution (median) | 0.9s | <60s | ✓ |

### Local Cost

- Cost per Successful Task: $0.000035

### Novel Efficiency Metrics (ATLAS Baselines)

- **Tokens per Watt-Hour:** 1,081
- **Tasks per Watt-Hour:** 9.54

*These are novel metrics establishing ATLAS V1 baselines.*

### Cloud API Cost Comparison

| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |
|----------|------------|------------|-------|---------------|
| claude-haiku | $0.0549 | $0.0097 | 5.7x | ✗ |
| claude-sonnet | $0.6590 | $0.0097 | 67.9x | ✓ |
| gpt-4o | $0.7846 | $0.0097 | 80.9x | ✓ |
| gpt-4o-mini | $0.0282 | $0.0097 | 2.9x | ✗ |

**Cost Efficiency Target: ≥30x cheaper than cloud APIs**
**Status: ✗ Target not met** (minimum ratio: 2.9x)

### CUSTOM pass@1

## Cost Analysis

### Summary

- Total Tasks: 100
- Successful Tasks: 66
- Total Tokens: 9,700
- Total Inference Time: 222.7s
- Total Wall Time: 224.2s
- Estimated Energy: 0.0089 kWh

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 1605.9 tasks/hr | ≥100 tasks/hr | ✓ |
| Time to Solution (median) | 1.5s | <60s | ✓ |

### Local Cost

- Cost per Successful Task: $0.000045

### Novel Efficiency Metrics (ATLAS Baselines)

- **Tokens per Watt-Hour:** 1,089
- **Tasks per Watt-Hour:** 7.41

*These are novel metrics establishing ATLAS V1 baselines.*

### Cloud API Cost Comparison

| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |
|----------|------------|------------|-------|---------------|
| claude-haiku | $0.0170 | $0.0030 | 5.7x | ✗ |
| claude-sonnet | $0.2037 | $0.0030 | 68.5x | ✓ |
| gpt-4o | $0.2425 | $0.0030 | 81.5x | ✓ |
| gpt-4o-mini | $0.0087 | $0.0030 | 2.9x | ✗ |

**Cost Efficiency Target: ≥30x cheaper than cloud APIs**
**Status: ✗ Target not met** (minimum ratio: 2.9x)

### HUMANEVAL pass@20

## Cost Analysis

### Summary

- Total Tasks: 164
- Successful Tasks: 164
- Total Tokens: 279,400
- Total Inference Time: 6376.1s
- Total Wall Time: 6420.1s
- Estimated Energy: 0.2550 kWh

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 92.0 tasks/hr | ≥100 tasks/hr | ✗ |
| Time to Solution (median) | 1.6s | <60s | ✓ |

### Local Cost

- Cost per Successful Task: $0.000519

### Novel Efficiency Metrics (ATLAS Baselines)

- **Tokens per Watt-Hour:** 1,095
- **Tasks per Watt-Hour:** 0.64

*These are novel metrics establishing ATLAS V1 baselines.*

### Cloud API Cost Comparison

| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |
|----------|------------|------------|-------|---------------|
| claude-haiku | $0.4889 | $0.0852 | 5.7x | ✗ |
| claude-sonnet | $5.8674 | $0.0852 | 68.9x | ✓ |
| gpt-4o | $6.9850 | $0.0852 | 82.0x | ✓ |
| gpt-4o-mini | $0.2515 | $0.0852 | 3.0x | ✗ |

**Cost Efficiency Target: ≥30x cheaper than cloud APIs**
**Status: ✗ Target not met** (minimum ratio: 3.0x)

### CUSTOM pass@20

## Cost Analysis

### Summary

- Total Tasks: 100
- Successful Tasks: 93
- Total Tokens: 291,028
- Total Inference Time: 6723.4s
- Total Wall Time: 6754.7s
- Estimated Energy: 0.2689 kWh

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 53.3 tasks/hr | ≥100 tasks/hr | ✗ |
| Time to Solution (median) | 1.8s | <60s | ✓ |

### Local Cost

- Cost per Successful Task: $0.000966

### Novel Efficiency Metrics (ATLAS Baselines)

- **Tokens per Watt-Hour:** 1,082
- **Tasks per Watt-Hour:** 0.35

*These are novel metrics establishing ATLAS V1 baselines.*

### Cloud API Cost Comparison

| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |
|----------|------------|------------|-------|---------------|
| claude-haiku | $0.5093 | $0.0898 | 5.7x | ✗ |
| claude-sonnet | $6.1116 | $0.0898 | 68.0x | ✓ |
| gpt-4o | $7.2757 | $0.0898 | 81.0x | ✓ |
| gpt-4o-mini | $0.2619 | $0.0898 | 2.9x | ✗ |

**Cost Efficiency Target: ≥30x cheaper than cloud APIs**
**Status: ✗ Target not met** (minimum ratio: 2.9x)


---

## Hardware Configuration

## Hardware Information

### GPU
- Model: NVIDIA GeForce RTX 5060 Ti
- VRAM: 15.9 GB
- Driver: 590.48.01
- CUDA: 13.1
- Power Draw: 11W

### CPU
- Model: AMD Ryzen 5 2600 Six-Core Processor
- Cores: 5

### System
- RAM: 13.4 GB
- OS: Linux 5.14.0-611.26.1.el9_7.x86_64
- Kernel: #1 SMP PREEMPT_DYNAMIC Sat Jan 17 05:14:35 EST 2026

### Software
- K3s: v1.34.3
- llama.cpp: N/A

### Model
- Name: Qwen3-14B-Q4_K_M.gguf
- Quantization: Q4_K_M
- Context Length: N/A

---

## Run Details

- **Run Directory:** `/home/isaac/ATLAS/benchmark/results/full_run_20260204_140715`
- **Log File:** `/home/isaac/ATLAS/benchmark/results/full_run_20260204_140715/full_benchmark.log`
- **Crash Log:** `/home/isaac/ATLAS/benchmark/results/full_run_20260204_140715/crash_log.json`
- **Report File:** `/home/isaac/ATLAS/benchmark/results/benchmark_report_20260204_140715.md`

### Individual Result Files

```
/home/isaac/ATLAS/benchmark/results/full_run_20260204_140715/
├── pass1/
│   ├── humaneval/
│   ├── mbpp/
│   └── custom/
└── passk/
    ├── humaneval_run1/
    ├── humaneval_run2/
    ├── humaneval_run3/
    ├── custom_run1/
    ├── custom_run2/
    └── custom_run3/
```

---

*Generated by ATLAS V1 Benchmark Suite*
