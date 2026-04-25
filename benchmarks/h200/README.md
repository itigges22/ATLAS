# H200 SXM Benchmark Runbook

One-shot guide to running the **ATLAS V3 pipeline** on a rented H200 SXM. Target total cost: under $100 at $3.99/hr.

**Baseline strategy:** we don't run baseline ourselves — Qwen's published numbers (bf16, their internal stack) are the reference. ATLAS runs on **AWQ-Q4 / vLLM** — identical to what users ship with. The delta vs Qwen's bf16 baseline is a **conservative lower bound** on what the pipeline adds, because ATLAS is running quantized against a full-precision reference. Keeps the Geometric Lens in-distribution (trained on 4096-dim hidden states) with no retraining required.

## Files in this directory

- `transfer_to_h200.sh` — rsync model, code, and current benchmark state to the rental (run locally before connecting)
- `launch_on_h200.sh` — on the H200: build vLLM container, start server with `--max-num-seqs 32`, kick off benchmarks
- `watchdog.sh` — on the H200: polls for completion markers and shuts the instance down when everything is done

## Before you start

You'll need:
- SSH access to the H200 instance (IP, user, key path)
- Confirmation the instance has:
  - At least 150GB disk free (container images + model + benchmark artifacts)
  - Podman or Docker with NVIDIA runtime
  - CUDA 12.8 available on the host (vLLM nightly requires 12.8+)

If the provider ships a pre-built PyTorch image, that's fine — we'll build the vLLM container ourselves inside a container.

## Step-by-step

### 1. Stop the 5060 Ti run (locally)

```bash
pkill -9 -f "run_full_baseline"
pkill -9 -f "v301_runner"
```

This saves the current state of `benchmarks/section_*/responses.jsonl` so we can resume on the H200.

### 2. Transfer everything to the H200

```bash
# Edit these at the top of the script first
cd /home/isaac/ATLAS
./benchmarks/h200/transfer_to_h200.sh
```

~12 GiB of AWQ shards + maybe 100MB of code and state. 15-30 min depending on upload bandwidth.

### 3. SSH to the H200 and launch

```bash
ssh <user>@<h200-ip>
cd ATLAS
./benchmarks/h200/launch_on_h200.sh
```

This builds the container (~5 min), starts vLLM on port 8000 with `--max-num-seqs 32`, smoke-tests it, and then runs baseline + ATLAS in sequence.

### 4. Let the watchdog manage shutdown

Separately (in another SSH session or tmux):

```bash
./benchmarks/h200/watchdog.sh
```

It polls for `results.json` in all expected benchmark dirs. When all are present:
1. Archives the benchmark results to `/tmp/atlas_results.tar.gz`
2. Prints the exact `rsync` command you'd run locally to pull results back
3. Runs `sudo shutdown -h now` to stop billing

### 5. Pull results back (locally)

```bash
rsync -avz <user>@<h200-ip>:/tmp/atlas_results.tar.gz ./
tar xzf atlas_results.tar.gz -C benchmarks/
```

## Expected runtime and cost

| Segment | Estimate |
|---|---|
| Setup (build container, transfer state) | ~45 min ($3) |
| Baseline remaining (MMLU-Pro + IFEval + GPQA + IFBench) | 8-12h ($32-48) |
| ATLAS V3 pipeline across 5 benchmarks | 15-20h ($60-80) |
| **Baseline + ATLAS total** | **~25-32h ($100-130)** |
| **Baseline only** | **~9-13h ($36-52)** |

To stay strictly under $100, do baseline only on the H200 and leave ATLAS for a later session or the 5060 Ti.

## If something breaks

- Container build fails: check `CUDA_ARCH` in `launch_on_h200.sh` — should be `90` for H100/H200 (Hopper).
- Server OOMs at parallel 16: drop to `--parallel 8 -c 131072` in the launch script.
- DeltaNet hang: already mitigated with `--no-cache-prompt --ctx-checkpoints 0` in the entrypoint.
- Instance dies mid-run: responses.jsonl is written incrementally with atomic appends. Re-run `./benchmarks/h200/launch_on_h200.sh` — runner's `find_completed()` skips finished tasks.
