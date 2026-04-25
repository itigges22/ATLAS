# ATLAS benchmark runner image (vLLM-based) for RunPod / Lambda / any cloud NVIDIA pod.
#
# Single-stage runtime: vLLM nightly + Python ATLAS runner + preloaded NLTK data + Lens weights.
#
# Inference backend:
#   Two vLLM instances run side by side from a single image:
#     - GEN  (port 8000): Qwen3.5-9B-AWQ, --reasoning-parser qwen3, prefix caching, max-num-seqs 32
#     - EMBED (port 8001): same model, --runner pooling --convert embed,
#       max-num-seqs 8 (4096-dim hidden states for the Lens)
#
# Model weights are PULLED on first run from HuggingFace (QuantTrio/Qwen3.5-9B-AWQ, ~12 GiB).
# Set DOWNLOAD_MODEL=0 and mount /workspace/models to skip the download.

FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Ubuntu 22.04 ships Python 3.10 by default.
# Real-time logs in `docker logs` / `kubectl logs` — without these, Python
# block-buffers when stdout is a pipe and progress lines stall.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 python3 python3-pip python3-dev \
        wget curl jq ca-certificates git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# vLLM 0.17.1 is the last known-working version for Qwen3.5 — v0.18.0 ships
# a regression where the engine core dies during initialization for Qwen3.5
# specifically (other Qwen variants are fine). Pin tightly until upstream
# fixes it; override via --build-arg if you've validated a newer release.
#   See https://github.com/vllm-project/vllm/issues/37749
#
# Range >=0.17.0,<0.18 is also acceptable — both 0.17.0 and 0.17.1 work.
ARG VLLM_SPEC="vllm==0.17.1"
RUN pip3 install --no-cache-dir -U \
        --pre \
        --index-url https://pypi.org/simple \
        --extra-index-url https://wheels.vllm.ai/nightly \
        "${VLLM_SPEC}"

# Qwen3.5 needs a transformers build that allows the partial_rotary_factor
# rope key. Install from the upstream commit the QuantTrio AWQ recipe was
# validated against, then patch modeling_rope_utils.py to accept that key.
#
# Patch is content-based, not line-number-based: we look for the existing
# `ignore_keys_at_rope_validation = ...` assignment that comes right before
# the rope-key-validation block and inject our addition right after it. If a
# future transformers commit re-arranges that file, the injection still
# lands at the right place. We fail loudly if the anchor line isn't found.
RUN pip3 install --no-cache-dir -U \
        "transformers @ git+https://github.com/huggingface/transformers.git@f2ba019" \
    && TF_FILE="$(python -m pip show transformers | awk -F': ' '/^Location:/{print $2}')/transformers/modeling_rope_utils.py" \
    && python3 -c "import sys, re; \
p = sys.argv[1]; \
src = open(p).read(); \
needle = 'ignore_keys_at_rope_validation'; \
assert needle in src, f'transformers patch anchor not found in {p}'; \
inject = '            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | {\"partial_rotary_factor\"}\n'; \
out = re.sub(r'(\\n[^\\n]*ignore_keys_at_rope_validation\\s*=\\s*[^\\n]+\\n)', r'\\1' + inject, src, count=1); \
assert out != src, 'patch made no changes'; \
open(p, 'w').write(out); \
print('patched ' + p)" "$TF_FILE"

# ATLAS runner deps (eval libs).
RUN pip3 install --no-cache-dir \
    langdetect nltk immutabledict absl-py emoji syllapy pyarrow numpy

# Geometric Lens service deps. CPU-only torch is enough — Lens doesn't need GPU
# (vLLM owns the GPU). xgboost loads gx_xgboost.pkl, fastapi+uvicorn serve the API.
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.10.0" \
    xgboost scikit-learn \
    fastapi uvicorn pydantic httpx pyyaml python-multipart tiktoken \
    redis tree-sitter tree-sitter-python

# Preload NLTK tokenizer data for IFEval/IFBench evaluators.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# ATLAS source.
WORKDIR /workspace/ATLAS
COPY benchmarks/ ./benchmarks/
COPY benchmark/ ./benchmark/
COPY geometric-lens/ ./geometric-lens/

# Download Geometric Lens model weights at BUILD time (~13 MB total).
RUN mkdir -p /workspace/ATLAS/geometric-lens/geometric_lens/models && \
    cd /workspace/ATLAS/geometric-lens/geometric_lens/models && \
    wget -q --show-progress \
        "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models/cost_field.pt" \
        -O cost_field.pt && \
    wget -q --show-progress \
        "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models/gx_xgboost.pkl" \
        -O gx_xgboost.pkl && \
    wget -q --show-progress \
        "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models/gx_weights.json" \
        -O gx_weights.json && \
    ls -lh cost_field.pt gx_xgboost.pkl gx_weights.json

# Default knobs. Override per-pod via -e flags.
ENV MODEL_NAME=QuantTrio/Qwen3.5-9B-AWQ \
    MODEL_PATH=/workspace/models/Qwen3.5-9B-AWQ \
    DOWNLOAD_MODEL=1 \
    GEN_PORT=8000 \
    EMBED_PORT=8001 \
    GEN_MAX_NUM_SEQS=32 \
    GEN_MAX_MODEL_LEN=32768 \
    GEN_GPU_MEM_UTIL=0.55 \
    EMBED_MAX_NUM_SEQS=8 \
    EMBED_MAX_MODEL_LEN=4096 \
    EMBED_GPU_MEM_UTIL=0.20 \
    BENCHMARK_PARALLEL=16 \
    ATLAS_LLM_PARALLEL=1 \
    ATLAS_PARALLEL_TASKS=16 \
    MODE=atlas_only \
    RESULT_TAR=/workspace/results/atlas_results.tar.gz \
    SHUTDOWN_ON_COMPLETE=0 \
    SKIP_SMOKE=0 \
    LLAMA_GEN_URL=http://localhost:8000 \
    LLAMA_EMBED_URL=http://localhost:8001 \
    RAG_API_URL=http://localhost:31144 \
    GEOMETRIC_LENS_ENABLED=true \
    LENS_PORT=31144 \
    OMP_NUM_THREADS=4

RUN mkdir -p /workspace/models /workspace/results /data/projects

COPY benchmarks/h200/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8001 31144

ENTRYPOINT ["/entrypoint.sh"]
