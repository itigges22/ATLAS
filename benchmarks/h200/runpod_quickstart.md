# RunPod Quickstart for ATLAS Benchmarking

## Build and push the image

From the ATLAS repo root (locally):

```bash
# Option A: use ghcr.io (free for public repos)
docker build \
    --build-arg CUDA_ARCH=90 \
    -f benchmarks/h200/Dockerfile \
    -t ghcr.io/itigges22/atlas-bench:h200 .
docker push ghcr.io/itigges22/atlas-bench:h200

# Option B: Docker Hub
docker build \
    --build-arg CUDA_ARCH=90 \
    -f benchmarks/h200/Dockerfile \
    -t itigges22/atlas-bench:h200 .
docker push itigges22/atlas-bench:h200
```

CUDA_ARCH values:
- `90` — H100, H200 (Hopper)
- `89` — RTX 40xx, L4 (Ada)
- `86` — A100 (Ampere)
- `120;121` — RTX 50xx, B100/B200 (Blackwell) — multi-arch builds work too: `"89;90"` for Ada+Hopper

Build time: ~3-6 min (pip-install of pinned `vllm==0.17.1` plus the content-based transformers patch — no native compilation). Image size: ~9-11 GiB (the vLLM nightly base ships with PyTorch + CUDA + Triton kernels; no model is baked in, the entrypoint pulls QuantTrio/Qwen3.5-9B-AWQ on first run unless `/workspace/models` is mounted).

## Deploy on RunPod

1. **Templates → New Template**
   - Container Image: `ghcr.io/itigges22/atlas-bench:h200` (or wherever you pushed)
   - Container Disk: 50 GB minimum (model is ~12 GiB AWQ + intermediate files)
   - Volume: Optional — attach a 20 GB persistent volume mounted at `/workspace/models` to cache the AWQ shards across pods
   - Environment Variables:
     ```
     MODEL_NAME=QuantTrio/Qwen3.5-9B-AWQ
     MODEL_PATH=/workspace/models/Qwen3.5-9B-AWQ
     DOWNLOAD_MODEL=1          # let entrypoint fetch from HF on first run; 0 if mounted
     # HF_TOKEN=hf_...          # only if pulling a gated repo
     GEN_MAX_NUM_SEQS=32       # vLLM concurrent slots (drop to 4-8 on consumer cards)
     GEN_MAX_MODEL_LEN=32768   # per-slot context window
     GEN_GPU_MEM_UTIL=0.55     # fraction of VRAM for the gen instance
     EMBED_MAX_NUM_SEQS=8
     EMBED_MAX_MODEL_LEN=4096
     EMBED_GPU_MEM_UTIL=0.20
     BENCHMARK_PARALLEL=16
     ATLAS_PARALLEL_TASKS=16
     MODE=atlas_only           # atlas_only | baseline_only | all
     SHUTDOWN_ON_COMPLETE=0    # 0 = stay alive after results (so you can rsync them)

     # Spot-resilience knobs (recommended for spot instances)
     SNAPSHOT_INTERVAL_SEC=900    # auto-snapshot every 15 min during sweep

     # OPTIONAL: HuggingFace dataset push (pod -> HF) as an extra durability
     # layer. The recommended path is the local pull-script — see step 4
     # below. Only enable HF push if you also want a public archive.
     # ATLAS_HF_DATASET=user/dataset
     # HF_TOKEN=hf_...
     ```
   - Exposed Ports: `8000/http` (vLLM gen) and `8001/http` (vLLM embed) — let you probe vLLM directly from the web terminal.

2. **Deploy → pick H200 SXM** (or H100, A100, whatever you want)

3. **Connect to pod** (web terminal or SSH) once it's `Running`. The container entrypoint starts automatically. Watch progress:
   ```bash
   tail -f /tmp/vllm-gen.log    # gen instance
   tail -f /tmp/vllm-embed.log  # embed instance
   tail -f /tmp/lens-service.log
   tail -f /workspace/results/logs/*/atlas_lcb.log

   # Watch the periodic snapshot loop
   ls -lt /workspace/results/atlas_results_*.tar.gz | head
   ```

4. **Pull results back to your local machine.** The container produces multiple tarballs at `/workspace/results/atlas_results_<RUN_ID>_<LABEL>.tar.gz`:
   - `*_preflight.tar.gz` — manifest only (pre-sweep, for debugging config)
   - `*_periodic.tar.gz` — every `SNAPSHOT_INTERVAL_SEC` during the sweep (kept rolling, last 5)
   - `*_baseline-done.tar.gz` / `*_atlas-done.tar.gz` — phase milestones
   - `*_sigterm.tar.gz` — emergency snapshot if the spot reclaims (you want this one)
   - `*_final.tar.gz` — the canonical end-of-run archive (`atlas_results_latest.tar.gz` symlinks to it)

   **Recommended: continuous rsync-pull from this machine.** Start it once when the pod boots and let it run alongside the sweep — every `SNAPSHOT_INTERVAL_SEC`, the latest tarball lands locally, surviving any subsequent spot reclaim:
   ```bash
   # Get the pod's SSH connection details from the RunPod web UI
   # ("Connect → SSH" gives you something like `ssh root@149.36.0.42 -p 12345`)

   # Continuous pull loop (every 5 min by default), auto-rehydrating any
   # *_final.tar.gz that lands:
   ./scripts/pull_pod_snapshots.sh root@<pod-host> -p <ssh-port> --rehydrate

   # One-shot pull (e.g., after the sweep finished and you just want the final):
   ./scripts/pull_pod_snapshots.sh root@<pod-host> -p <ssh-port> --once
   ```

   Why pull (vs. `runpodctl receive` or HF push):
   - **No SSH server on your laptop** — the pod is the SSH server, your machine is the client.
   - **Works through NAT / dynamic IP / coffee-shop wifi** — outbound SSH only.
   - **rsync is incremental** — re-runs are cheap, partial transfers resume.
   - **Survives pod reclaim** — whatever was on disk at SIGTERM-snapshot time is already local.

   Alternative one-off pull via runpodctl (if SSH isn't set up):
   ```bash
   runpodctl receive <pod-id>:/workspace/results/atlas_results_latest.tar.gz ./
   ```

   Optional: `ATLAS_HF_DATASET` + `HF_TOKEN` enables a *secondary* push from the pod to a HuggingFace dataset. Not needed for the local-report flow — only useful if you also want a public archive of the run.

5. **Build the report locally.** Once any tarball lands on your machine:
   ```bash
   # Lay it out under ./rehydrated/<RUN_ID>/
   ./scripts/rehydrate_results.sh ./atlas_results_latest.tar.gz

   # Render baseline + ATLAS comparison report
   python3 scripts/build_v31_report.py \
       --baseline rehydrated/<baseline_run_id> \
       --atlas    rehydrated/<atlas_run_id> \
       --out      reports/V3.1_BENCHMARKS.md
   ```
   The report includes per-benchmark deltas with 95% CIs, extraction-failure notes, and the full reproducibility manifest fingerprint (git SHA, vLLM/transformers versions, model SHA256, hostname).

6. **Stop the pod** on the RunPod dashboard to stop billing.

## Budget estimate (H200 SXM, $3.99/hr)

| Phase | Time | Cost |
|-------|------|------|
| Pod cold boot + model download | ~5 min | $0.35 |
| Container startup + smoke test | ~3 min | $0.20 |
| ATLAS V3 pipeline, all 5 benchmarks | 18-25h | $72-100 |
| **Total** | **~19-26h** | **~$73-100** |

Tips to stay under $100:
- Set `SHUTDOWN_ON_COMPLETE=1` if you can't be around to stop the pod manually
- Use spot pricing if available — the snapshot pipeline (manifest, periodic auto-snapshot, SIGTERM trap) is built around exactly this case. If a spot reclaim hits, you lose at most `SNAPSHOT_INTERVAL_SEC` seconds of work, plus whatever the SIGTERM-handler's emergency snapshot couldn't capture in the ~1-2 min between SIGTERM and SIGKILL.
- Pre-populate a network volume with the model so `DOWNLOAD_MODEL=0` saves ~5 min
- Start `scripts/pull_pod_snapshots.sh` on this machine the moment the pod is reachable so partials are already local before any reclaim

## Troubleshooting

- **"no nvidia-smi"**: pod doesn't have GPU. Pick a GPU template.
- **vLLM OOM**: drop `GEN_MAX_NUM_SEQS` to 8 or 4, drop `GEN_MAX_MODEL_LEN` to 16384, or lower `GEN_GPU_MEM_UTIL` to 0.45. On a single 16 GB card, set `SKIP_EMBED=1` in the env to skip the embed instance entirely (the entrypoint will also disable the Lens, since C(x)/G(x) need embeddings — V3 falls back to sandbox-only verification).
- **Smoke test fails**: check `/tmp/vllm-gen.log` (and `/tmp/vllm-embed.log`) for the actual error. Common ones: AWQ download failed (set HF_TOKEN), model not found at MODEL_PATH, or insufficient CUDA driver (need 12.8+).
- **Slow download**: RunPod has cached HuggingFace hits — first download might be fast if the node has seen it before.
- **Container exits immediately**: check Pod Logs. Most likely the model failed to download. Set `DOWNLOAD_MODEL=0` and mount a volume instead.
