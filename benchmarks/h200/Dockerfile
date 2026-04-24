# ATLAS benchmark runner image for RunPod / Lambda / any cloud NVIDIA pod.
#
# Two stages:
#  1. Build llama.cpp from source for Hopper (H100/H200). CUDA_ARCH arg
#     lets you retarget (89 for RTX 40xx, 90 for Hopper, 120;121 for Blackwell).
#  2. Slim runtime image that contains llama-server + Python ATLAS runner.
#
# Default entrypoint runs the ATLAS V3 pipeline on Q6_K weights.
# Baseline numbers are cited from Qwen's published model card.
#
# Model weights are NOT baked into the image — either:
#   - Mount a RunPod network volume at /workspace/models/
#   - Or let the entrypoint download on first run (set DOWNLOAD_MODEL=1)

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

# Python + utils
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        wget curl jq ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Python packages for the ATLAS runner (stdlib-heavy, minimal deps)
RUN pip install --no-cache-dir --break-system-packages \
    langdetect nltk immutabledict absl-py emoji syllapy pyarrow numpy

# llama-server binary from builder
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# ATLAS code — copy benchmarks/ and benchmark/ dirs into /workspace/ATLAS/
WORKDIR /workspace/ATLAS
COPY benchmarks/ ./benchmarks/
COPY benchmark/ ./benchmark/

# Default paths (override at runtime with -e or RunPod env vars)
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
    SHUTDOWN_ON_COMPLETE=0

RUN mkdir -p /workspace/models /workspace/results

COPY benchmarks/h200/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# llama-server health port
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
