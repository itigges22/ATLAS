# ATLAS V2 Troubleshooting Guide

This guide covers common issues and their solutions for the V2 deployment.

---

## Quick Diagnostics

```bash
# Check pod status
kubectl get pods -n atlas

# Check pod events
kubectl describe pod <pod-name> -n atlas

# View pod logs
kubectl logs <pod-name> -n atlas

# Follow logs in real-time
kubectl logs -f <pod-name> -n atlas

# Check service endpoints
kubectl get svc -n atlas

# Check persistent volumes
kubectl get pvc -n atlas
```

---

## V2-Specific Issues

### mlock Failure (Resolved)

**Symptom:** llama-server logs report `Cannot allocate memory` when the `--mlock` flag is active.

**Fix (applied):** Create a systemd override for K3s to set unlimited memlock:

```bash
sudo mkdir -p /etc/systemd/system/k3s.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/k3s.service.d/memlock.conf
[Service]
LimitMEMLOCK=infinity
EOF
sudo systemctl daemon-reload
sudo systemctl restart k3s
```

If you see this error on a fresh install, apply the override above and restart K3s.

---

### Speculative Decoding Slot 1 Failure (Resolved)

**Symptom:** Slot 0 works with speculative decoding but slot 1 fails with `failed to create draft context`.

**Root cause:** The draft model KV cache was using f16 (2,240 MiB per slot), leaving insufficient VRAM for a second draft context.

**Fix (applied):** The entrypoint now passes `-ctkd q4_0 -ctvd q4_0` to quantize the draft KV cache, reducing it from 2,240 MiB to 630 MiB per slot. Both slots now initialize with speculative decoding.

If you see this error, ensure `KV_CACHE_TYPE` is set in the llama-server deployment (the entrypoint applies it to both main and draft KV caches via `-ctk`/`-ctv`/`-ctkd`/`-ctvd`).

---

### Dashboard Not Loading (Resolved)

**Symptom:** The atlas-dashboard deployment crashes with a Jinja2 `UndefinedError` on `daily_stats.total_attempts`.

**Fix (applied):** Added `|default(0)` filters to the dashboard template for Redis hash fields that may not exist when no tasks have run that day.

---

### Per-Slot Context Truncation

**Symptom:** llama-server logs contain the warning `n_ctx_seq (20480) < n_ctx_train (40960)`.

**Root cause:** The `--ctx-size 40960` setting is divided equally across `--parallel 2` slots, giving each slot 20,480 tokens of context. The model was trained with a 40,960-token context window, so the server warns that each slot has less context than the model was trained for.

**Impact:** Each individual request is limited to approximately 20,000 tokens of context (prompt + generation), not the full 40,960. This is sufficient for most benchmark tasks and RAG queries but will truncate very long prompts.

**Workaround:** To give a single slot the full 40,960 context, set `ATLAS_PARALLEL_SLOTS=1`. This eliminates concurrent request handling but removes the context split.

---

### GPU Memory Pressure

**Symptom:** Inference becomes slow, requests time out, or CUDA out-of-memory errors appear.

**Monitor VRAM usage:**

```bash
kubectl exec deployment/llama-server -n atlas -- nvidia-smi
```

Typical V2 VRAM usage with Qwen3-14B-Q4_K_M + Qwen3-0.6B-Q8_0 draft + 2 slots:

```
Memory: ~13,635 / 16,311 MiB (83.6%)
```

There is little headroom. Do not attempt to load additional models or increase context length without first checking available VRAM.

**Mitigation options:**

1. Reduce context length: `ATLAS_CONTEXT_LENGTH=32768` (saves ~1-2 GB)
2. Reduce to 1 parallel slot: `ATLAS_PARALLEL_SLOTS=1` (saves one KV cache allocation)
3. Disable speculative decoding: unset `DRAFT_MODEL` (saves ~600 MiB)
4. Use a more aggressively quantized model (Q3_K_M instead of Q4_K_M)

---

## Installation Issues

### Installer fails with "command not found"

**Symptom:** `./scripts/install.sh: command not found`

**Solution:**

```bash
chmod +x scripts/install.sh
bash scripts/install.sh
```

### "atlas.conf not found"

**Symptom:** Installer complains about missing configuration.

**Solution:**

```bash
cp atlas.conf.example atlas.conf
vim atlas.conf
```

### kubectl not configured

**Symptom:** `KUBECONFIG not set` or connection refused.

**Solution:**

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG=~/.kube/config
```

---

## GPU Issues

### GPU not detected

**Symptom:** `nvidia.com/gpu: 0` in node allocatable resources.

**Check each layer of the stack:**

1. **NVIDIA driver:**
   ```bash
   nvidia-smi
   # If missing: sudo dnf install -y nvidia-driver nvidia-driver-cuda && sudo reboot
   ```

2. **NVIDIA device plugin:**
   ```bash
   kubectl get pods -n kube-system | grep nvidia
   # If missing:
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
   ```

3. **Containerd NVIDIA runtime:**
   ```bash
   # Verify the containerd config template exists
   ls /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
   ```
   See [SETUP.md](SETUP.md) for the full containerd template.

### CUDA out of memory

**Symptom:** `CUDA error: out of memory`

**Solutions in order of impact:**

1. Reduce GPU layers: `ATLAS_GPU_LAYERS=80`
2. Use a smaller quantization: switch from Q4_K_M to Q3_K_M
3. Disable speculative decoding: `ATLAS_DRAFT_MODEL=""`
4. Reduce context length: `ATLAS_CONTEXT_LENGTH=16384`
5. Reduce parallel slots: `ATLAS_PARALLEL_SLOTS=1`

---

## Service Issues

### Pod stuck in Pending

**Symptom:** Pod shows `Pending` status indefinitely.

**Debug:**

```bash
kubectl describe pod <pod-name> -n atlas
```

**Common causes:**

1. **Insufficient resources:** Reduce resource requests in atlas.conf. Check node capacity with `kubectl describe node`.
2. **PVC not bound:** Check `kubectl get pvc -n atlas`. If Pending, verify the storage class exists: `kubectl get sc`.
3. **GPU not available:** See [GPU not detected](#gpu-not-detected).

### Pod in CrashLoopBackOff

**Symptom:** Pod repeatedly crashes and restarts.

**Debug:**

```bash
kubectl logs <pod-name> -n atlas
kubectl logs <pod-name> -n atlas --previous
kubectl describe pod <pod-name> -n atlas
```

**Common causes:**

1. **Model file not found:** Verify the model path matches atlas.conf and the hostPath mount exists on disk.
2. **Configuration error:** Check for typos in atlas.conf or invalid YAML in manifests.
3. **Port conflict:** Check if the NodePort is already in use by another service.

### Service returning 502/503

**Symptom:** API calls return bad gateway or service unavailable.

**Debug:**

```bash
# Check if pods are ready
kubectl get pods -n atlas

# Test health endpoint from inside the cluster
kubectl exec -it <pod-name> -n atlas -- curl localhost:8000/health

# Verify service endpoints are populated
kubectl get endpoints -n atlas
```

**Common cause:** llama-server takes 30-60 seconds to load the model on startup. The first request after a pod restart will be slow (cold KV cache warmup).

---

## Model Issues

### Model fails to load

**Symptom:** `error loading model` in llama-server logs.

**Check:**

```bash
# Verify file exists on the host
ls -la /opt/atlas/models/Qwen3-14B-Q4_K_M.gguf  # adjust to your ATLAS_MODELS_DIR

# Verify file is visible inside the pod
kubectl exec deployment/llama-server -n atlas -- ls -la /models/

# Verify filename matches config
grep ATLAS_MAIN_MODEL atlas.conf
```

### Slow inference

**Symptom:** Responses take significantly longer than expected.

**Possible causes and solutions:**

1. **Cold KV cache:** The first request after model load is always slow. Send a warmup request before benchmarking.
2. **Flash attention disabled:** Ensure `ATLAS_FLASH_ATTENTION=true`.
3. **Partial GPU offload:** Set `ATLAS_GPU_LAYERS=99` to offload all layers.
4. **Speculative decoding not active:** Check llama-server logs for `Speculative decoding enabled`. If slot 1 shows draft context failure, only slot 0 gets spec decode benefit.
5. **Qwen3 thinking mode:** At temperature=0, Qwen3 engages thinking mode with `<think>` tags that consume 8000+ tokens before generating the actual answer. Use the `/nothink` template mode or set `max_tokens` to at least 16384.

---

## RAG Issues

### Project sync fails

**Symptom:** Error during codebase sync to PageIndex.

**Debug:**

```bash
# Check rag-api logs
kubectl logs deployment/rag-api -n atlas

# Verify llama-server is healthy (needed for tree search and embeddings)
curl http://localhost:32735/health
```

**Common causes:**

1. **llama-server not ready:** PageIndex tree search requires the LLM for summarization. Wait for llama-server to finish loading.
2. **Project too large:** Reduce project size or increase `MAX_FILES`.
3. **Disk full:** Check available space on the projects volume.

### Poor retrieval quality

**Symptom:** Retrieved context is not relevant to the query.

**Solutions:**

1. **Increase TOP_K:** Set `TOP_K=30` in the rag-api deployment to retrieve more candidates.
2. **Re-sync project:** Delete the project and re-upload to rebuild the PageIndex tree and BM25 index.
3. **Check routing:** If `ROUTING_ENABLED=true`, the confidence router may be selecting FAST_PATH (k=1) for queries that need more context. Check `route_decisions.jsonl` telemetry.

---

## Memory Issues

### Node running out of memory

**Symptom:** Pods evicted or OOMKilled.

**Solutions:**

1. **Reduce memory requests:**
   ```bash
   ATLAS_LLAMA_MEMORY_REQUEST="4Gi"
   ATLAS_SERVICE_MEMORY_REQUEST="256Mi"
   ```

2. **Disable unused services:**
   ```bash
   ATLAS_ENABLE_DASHBOARD=false
   ATLAS_ENABLE_TRAINING=false
   ```

3. **Enable swap (not recommended for production):**
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Network Issues

### Cannot connect to service

**Symptom:** Connection refused or timeout when accessing a NodePort.

**Debug:**

```bash
# Check services exist
kubectl get svc -n atlas

# Check endpoints are populated
kubectl get endpoints -n atlas

# Test from inside the cluster
kubectl run test --rm -it --image=busybox -n atlas -- wget -qO- http://rag-api:8001/health

# Check firewall
sudo firewall-cmd --list-all
```

### NodePort not accessible

**Symptom:** Cannot reach service from outside the cluster.

**Solutions:**

1. **Open firewall port:**
   ```bash
   sudo firewall-cmd --permanent --add-port=32735/tcp
   sudo firewall-cmd --permanent --add-port=31144/tcp
   sudo firewall-cmd --reload
   ```

2. **Verify correct IP:**
   ```bash
   kubectl get nodes -o wide
   ```

3. **Verify NodePort assignment:**
   ```bash
   kubectl get svc -n atlas -o wide
   ```

---

## Collecting Logs

### Per-service logs

```bash
# Current logs
kubectl logs deployment/llama-server -n atlas
kubectl logs deployment/rag-api -n atlas

# Previous container logs (after a restart)
kubectl logs deployment/llama-server -n atlas --previous

# Follow logs in real-time
kubectl logs -f deployment/rag-api -n atlas
```

### System logs

```bash
# K3s logs
sudo journalctl -u k3s

# NVIDIA driver messages
dmesg | grep nvidia
```

### Export all logs for debugging

```bash
mkdir -p logs
for pod in $(kubectl get pods -n atlas -o name); do
  kubectl logs -n atlas $pod > logs/$(basename $pod).log 2>&1
done
```

---

## Getting Help

If you cannot resolve an issue:

1. Check existing GitHub issues.
2. Open a new issue with:
   - ATLAS version or commit hash (`git rev-parse HEAD`)
   - Output of `kubectl get pods -n atlas`
   - Relevant pod logs
   - Your `atlas.conf` (remove any secrets)
   - Steps to reproduce the problem
