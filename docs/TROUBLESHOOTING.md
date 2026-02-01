# ATLAS Troubleshooting Guide

This guide covers common issues and their solutions.

## Quick Diagnostics

```bash
# Check pod status
kubectl get pods

# Check pod events
kubectl describe pod <pod-name>

# View pod logs
kubectl logs <pod-name>

# Follow logs in real-time
kubectl logs -f <pod-name>

# Check service endpoints
kubectl get svc

# Check persistent volumes
kubectl get pvc
```

## Common Issues

### Installation Issues

#### Installer fails with "command not found"

**Symptom**: `./scripts/install.sh: command not found`

**Solution**:
```bash
# Make script executable
chmod +x scripts/install.sh

# Or run directly
bash scripts/install.sh
```

#### "atlas.conf not found"

**Symptom**: Installer complains about missing configuration

**Solution**:
```bash
# Copy example configuration
cp atlas.conf.example atlas.conf

# Edit with your settings
vim atlas.conf
```

#### kubectl not configured

**Symptom**: `KUBECONFIG not set` or connection refused

**Solution**:
```bash
# Copy K3s config
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
chmod 600 ~/.kube/config

# Or set KUBECONFIG
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

---

### GPU Issues

#### GPU not detected

**Symptom**: `nvidia.com/gpu: 0` in node resources

**Causes and Solutions**:

1. **NVIDIA driver not installed**
   ```bash
   # Check driver
   nvidia-smi

   # Install driver (RHEL/Rocky)
   sudo dnf install -y nvidia-driver nvidia-driver-cuda
   sudo reboot
   ```

2. **NVIDIA device plugin not installed**
   ```bash
   # Install device plugin
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

   # Verify
   kubectl get pods -n kube-system | grep nvidia
   ```

3. **Container runtime not configured for GPU**
   ```bash
   # For containerd (K3s default), check config
   sudo cat /etc/containerd/config.toml

   # Ensure nvidia runtime is configured
   ```

#### CUDA out of memory

**Symptom**: `CUDA error: out of memory`

**Solutions**:

1. **Reduce GPU layers**
   ```bash
   # In atlas.conf
   ATLAS_GPU_LAYERS=80  # Reduce from 99
   ```

2. **Use smaller model or quantization**
   ```bash
   # Use Q4 instead of Q6
   ATLAS_MAIN_MODEL="model-q4_k_m.gguf"
   ```

3. **Disable speculative decoding**
   ```bash
   ATLAS_DRAFT_MODEL=""
   ATLAS_ENABLE_SPECULATIVE=false
   ```

4. **Reduce context length**
   ```bash
   ATLAS_CONTEXT_LENGTH=8192  # Reduce from 16384
   ```

---

### Service Issues

#### Pod stuck in Pending

**Symptom**: Pod shows `Pending` status indefinitely

**Check events**:
```bash
kubectl describe pod <pod-name>
```

**Common causes**:

1. **Insufficient resources**
   - Reduce resource requests in atlas.conf
   - Check node capacity: `kubectl describe node`

2. **PVC not bound**
   ```bash
   kubectl get pvc
   # If PVC is Pending, check storage class
   kubectl get sc
   ```

3. **Missing GPU resource**
   - See [GPU not detected](#gpu-not-detected)

#### Pod in CrashLoopBackOff

**Symptom**: Pod repeatedly crashes and restarts

**Debug steps**:
```bash
# Check logs
kubectl logs <pod-name>

# Check previous container logs
kubectl logs <pod-name> --previous

# Check events
kubectl describe pod <pod-name>
```

**Common causes**:

1. **Model file not found**
   - Verify model path in atlas.conf
   - Check hostPath mount exists

2. **Configuration error**
   - Check for typos in atlas.conf
   - Validate JSON/YAML syntax in manifests

3. **Port conflict**
   - Check if port is already in use
   - Change NodePort values in atlas.conf

#### Service returning 502/503

**Symptom**: API returns bad gateway or service unavailable

**Solutions**:

1. **Service not ready**
   ```bash
   # Wait for all pods to be ready
   kubectl get pods -w
   ```

2. **Health check failing**
   ```bash
   # Check health endpoint directly
   kubectl exec -it <pod-name> -- curl localhost:8000/health
   ```

3. **Wrong service port**
   ```bash
   # Verify service endpoints
   kubectl get endpoints
   ```

---

### Model Issues

#### Model fails to load

**Symptom**: `error loading model` in llama-server logs

**Solutions**:

1. **Verify model file**
   ```bash
   # Check file exists and is readable
   ls -la /path/to/model.gguf

   # Check file size (should be several GB)
   du -h /path/to/model.gguf
   ```

2. **Verify mount in pod**
   ```bash
   kubectl exec -it llama-server-xxx -- ls -la /models/
   ```

3. **Check model filename matches config**
   ```bash
   grep ATLAS_MAIN_MODEL atlas.conf
   ```

#### Slow inference

**Symptom**: Responses take too long

**Solutions**:

1. **Enable flash attention**
   ```bash
   ATLAS_FLASH_ATTENTION=true
   ```

2. **Increase GPU layers**
   ```bash
   ATLAS_GPU_LAYERS=99
   ```

3. **Use speculative decoding**
   ```bash
   ATLAS_DRAFT_MODEL="smaller-model.gguf"
   ATLAS_ENABLE_SPECULATIVE=true
   ```

4. **Reduce context length if not needed**
   ```bash
   ATLAS_CONTEXT_LENGTH=8192
   ```

---

### Memory Issues

#### Node running out of memory

**Symptom**: Pods evicted, OOMKilled

**Solutions**:

1. **Reduce memory requests**
   ```bash
   ATLAS_LLAMA_MEMORY_REQUEST="4Gi"
   ATLAS_SERVICE_MEMORY_REQUEST="256Mi"
   ```

2. **Enable swap (not recommended for production)**
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Disable unused services**
   ```bash
   ATLAS_ENABLE_DASHBOARD=false
   ATLAS_ENABLE_TRAINING=false
   ```

#### Qdrant running out of storage

**Symptom**: Vector indexing fails

**Solutions**:

1. **Increase PVC size**
   ```bash
   ATLAS_PVC_QDRANT_SIZE="100Gi"
   ```

2. **Clean old projects**
   ```bash
   # Delete old vector collections
   curl -X DELETE http://localhost:6333/collections/old_project
   ```

---

### Network Issues

#### Cannot connect to service

**Symptom**: Connection refused or timeout

**Debug steps**:
```bash
# Check service exists
kubectl get svc

# Check endpoints exist
kubectl get endpoints <service-name>

# Test from inside cluster
kubectl run test --rm -it --image=busybox -- wget -qO- http://service:port/health

# Check firewall
sudo firewall-cmd --list-all
```

#### NodePort not accessible

**Symptom**: Cannot reach service from outside cluster

**Solutions**:

1. **Open firewall port**
   ```bash
   sudo firewall-cmd --permanent --add-port=30000/tcp
   sudo firewall-cmd --reload
   ```

2. **Check correct IP**
   ```bash
   kubectl get nodes -o wide
   ```

3. **Verify NodePort service**
   ```bash
   kubectl get svc -o wide
   ```

---

### Authentication Issues

#### Invalid API key

**Symptom**: 401 Unauthorized

**Solutions**:

1. **Verify key in database**
   - Log into API Portal
   - Check key is active (not revoked)
   - Check key hasn't expired

2. **Check rate limit not exceeded**
   - View usage in dashboard
   - Wait for rate limit window to reset

3. **Verify key format**
   - Keys should start with `sk-`
   - Check for trailing whitespace

#### JWT expired

**Symptom**: Session expired, logged out

**Solutions**:

1. **Increase expiry**
   ```bash
   ATLAS_JWT_EXPIRY_HOURS=168  # 1 week
   ```

2. **Re-login to get new token**

---

### RAG Issues

#### Project sync fails

**Symptom**: Error during codebase sync

**Solutions**:

1. **Check embedding service**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Check Qdrant**
   ```bash
   curl http://localhost:6333/health
   ```

3. **Reduce project size**
   - Exclude large directories
   - Use .gitignore patterns

#### Poor retrieval quality

**Symptom**: Retrieved context not relevant

**Solutions**:

1. **Increase top-k**
   ```bash
   ATLAS_RAG_TOP_K=30
   ```

2. **Adjust chunk size**
   ```bash
   ATLAS_RAG_CHUNK_SIZE=80
   ATLAS_RAG_CHUNK_OVERLAP=30
   ```

3. **Re-sync project**
   - Delete project and re-upload

---

## Getting Logs

### All pod logs

```bash
# Current logs
kubectl logs <pod-name>

# Previous container logs (after restart)
kubectl logs <pod-name> --previous

# Follow logs
kubectl logs -f <pod-name>

# Logs from all pods with label
kubectl logs -l app=rag-api
```

### System logs

```bash
# K3s logs
sudo journalctl -u k3s

# containerd logs
sudo journalctl -u containerd

# NVIDIA driver logs
dmesg | grep nvidia
```

### Export logs for debugging

```bash
# Export all ATLAS logs
for pod in $(kubectl get pods -o name); do
  kubectl logs $pod > logs/$(basename $pod).log
done
```

## Getting Help

If you can't resolve an issue:

1. Check existing GitHub issues
2. Open a new issue with:
   - ATLAS version/commit
   - Output of `kubectl get pods`
   - Relevant pod logs
   - Your atlas.conf (remove secrets)
   - Steps to reproduce
