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

Build time: ~5-10 min (compiles llama.cpp from source). Image size: ~3GB (no model baked in).

## Deploy on RunPod

1. **Templates → New Template**
   - Container Image: `ghcr.io/itigges22/atlas-bench:h200` (or wherever you pushed)
   - Container Disk: 50 GB minimum (for model + intermediate files)
   - Volume: Optional — attach a 20 GB persistent volume mounted at `/workspace/models` to cache the model across pods
   - Environment Variables:
     ```
     MODEL_PATH=/workspace/models/Qwen3.5-9B-Q6_K.gguf
     DOWNLOAD_MODEL=1          # let entrypoint fetch on first run, or set to 0 if mounted
     SERVER_PARALLEL=16        # drop to 4 on consumer cards, keep at 16 on H200
     SERVER_CONTEXT=262144     # = SERVER_PARALLEL × 16384 for a 16K/slot ctx
     BENCHMARK_PARALLEL=16
     ATLAS_PARALLEL_TASKS=16
     MODE=atlas_only           # atlas_only | baseline_only | all
     SHUTDOWN_ON_COMPLETE=0    # 0 = stay alive after results (so you can rsync them)
     ```
   - Exposed Ports: `8000/http` (optional — lets you hit the llama-server from the web terminal)

2. **Deploy → pick H200 SXM** (or H100, A100, whatever you want)

3. **Connect to pod** (web terminal or SSH) once it's `Running`. The container entrypoint starts automatically. Watch progress:
   ```bash
   tail -f /tmp/llama-server.log
   # and
   tail -f /workspace/results/logs/*/atlas_lcb.log
   ```

4. **When you see `Done. Results tarball: /workspace/results/atlas_results.tar.gz`**:
   ```bash
   # From your local machine:
   runpodctl receive <pod-id>:/workspace/results/atlas_results.tar.gz ./
   # or SSH + rsync
   ```

5. **Stop the pod** on the RunPod dashboard to stop billing.

## Budget estimate (H200 SXM, $3.99/hr)

| Phase | Time | Cost |
|-------|------|------|
| Pod cold boot + model download | ~5 min | $0.35 |
| Container startup + smoke test | ~3 min | $0.20 |
| ATLAS V3 pipeline, all 5 benchmarks | 18-25h | $72-100 |
| **Total** | **~19-26h** | **~$73-100** |

Tips to stay under $100:
- Set `SHUTDOWN_ON_COMPLETE=1` if you can't be around to stop the pod manually
- Use spot pricing if available
- Pre-populate a network volume with the model so `DOWNLOAD_MODEL=0` saves ~5 min

## Troubleshooting

- **"no nvidia-smi"**: pod doesn't have GPU. Pick a GPU template.
- **llama-server OOM**: drop `SERVER_PARALLEL` to 8 or 4, recalculate `SERVER_CONTEXT` (= SERVER_PARALLEL × 16384).
- **Smoke test fails**: check `/tmp/llama-server.log` for missing flags or wrong CUDA arch. You may need to rebuild the image with the correct `CUDA_ARCH`.
- **Slow download**: RunPod has cached HuggingFace hits — first download might be fast if the node has seen it before.
- **Container exits immediately**: check Pod Logs. Most likely the model failed to download. Set `DOWNLOAD_MODEL=0` and mount a volume instead.
