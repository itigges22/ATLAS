# ATLAS benchmark runner image for RunPod / Lambda / any cloud NVIDIA pod.
#
# Two stages:
#  1. Build llama.cpp from source for the target GPU (CUDA_ARCH arg).
#  2. Slim runtime image with llama-server + Python ATLAS runner + preloaded NLTK data.
#
# Model weights are NOT baked in — mount a volume at /workspace/models or
# set DOWNLOAD_MODEL=1 to let the entrypoint fetch on first run.

# ---------- stage 1: build llama.cpp ----------
FROM docker.io/nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=90

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git cmake gcc g++ ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp && \
    cd /llama.cpp && \
    # ATLAS patch: draft model should not inherit --embeddings from main.
    # No-op if line doesn't exist in this llama.cpp version.
    sed -i '/auto params_dft = params_base;/a\        params_dft.embedding = false;  // ATLAS: draft never needs embeddings' \
        tools/server/server-context.cpp 2>/dev/null || true && \
    cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" && \
    cmake --build build --config Release -j"$(nproc)" --target llama-server

# ---------- stage 2: runtime image ----------
FROM docker.io/nvidia/cuda:12.9.0-runtime-ubuntu22.04

# Ubuntu 22.04 ships Python 3.10 by default — use it directly (no deadsnakes PPA).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 python3 python3-pip \
        wget curl jq ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Python packages for the ATLAS runner. Python 3.10 on 22.04 is pre-PEP-668
# so no --break-system-packages flag needed.
RUN pip3 install --no-cache-dir \
    langdetect nltk immutabledict absl-py emoji syllapy pyarrow numpy

# Geometric Lens service deps (torch CPU is ~200MB; ATLAS V3 needs C(x) and G(x)).
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.10.0" \
    xgboost scikit-learn \
    fastapi uvicorn pydantic httpx pyyaml python-multipart tiktoken \
    redis tree-sitter tree-sitter-python

# Preload NLTK tokenizer data — IFEval/IFBench evaluators call
# nltk.word_tokenize() which needs 'punkt' and 'punkt_tab'. Baking this in
# avoids a runtime download attempt that may fail silently.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# llama-server binary from builder
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# ATLAS code — copy benchmarks/, benchmark/, and geometric-lens/ into the image
WORKDIR /workspace/ATLAS
COPY benchmarks/ ./benchmarks/
COPY benchmark/ ./benchmark/
COPY geometric-lens/ ./geometric-lens/

# Download Geometric Lens model weights from HuggingFace at BUILD time so the
# image is self-contained. Total ~10MB (C(x) MLP + G(x) XGBoost) — small enough
# to bake in. Placed in the path the service expects.
RUN mkdir -p /workspace/ATLAS/geometric-lens/geometric_lens/models && \
    cd /workspace/ATLAS/geometric-lens/geometric_lens/models && \
    wget -q --show-progress \
        "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models/cost_field.pt" \
        -O cost_field.pt && \
    wget -q --show-progress \
        "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models/gx_xgboost.pkl" \
        -O gx_xgboost.pkl && \
    ls -lh cost_field.pt gx_xgboost.pkl

# Default paths and knobs (override at runtime)
ENV MODEL_PATH=/workspace/models/Qwen3.5-9B-Q6_K.gguf \
    MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf" \
    DOWNLOAD_MODEL=0 \
    SERVER_PORT=8000 \
    SERVER_PARALLEL=16 \
    SERVER_CONTEXT=262144 \
    BENCHMARK_PARALLEL=16 \
    ATLAS_LLM_PARALLEL=1 \
    ATLAS_PARALLEL_TASKS=16 \
    MODE=atlas_only \
    RESULT_TAR=/workspace/results/atlas_results.tar.gz \
    SHUTDOWN_ON_COMPLETE=0 \
    SKIP_SMOKE=0 \
    LLAMA_URL=http://localhost:8000 \
    RAG_API_URL=http://localhost:31144 \
    GEOMETRIC_LENS_ENABLED=true \
    LENS_PORT=31144

RUN mkdir -p /workspace/models /workspace/results /data/projects

COPY benchmarks/h200/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 31144

ENTRYPOINT ["/entrypoint.sh"]
