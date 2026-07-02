FROM docker.io/nvidia/cuda:12.9.0-devel-rockylinux9 AS builder
RUN dnf install -y git cmake gcc-c++ && dnf clean all
# Pin llama.cpp to a known-good SHA so the PC-202 patch applies cleanly.
# Keep aligned with Dockerfile.v31 / Dockerfile.rocm / Dockerfile.vulkan.
ARG LLAMA_CPP_REV=2e97c5f96f9fe2bb26f794a348e05d7a1c74baa1
RUN git init /llama.cpp && \
    cd /llama.cpp && \
    git remote add origin https://github.com/ggml-org/llama.cpp && \
    git fetch --depth 1 origin ${LLAMA_CPP_REV} && \
    git checkout FETCH_HEAD && \
    # ATLAS patch: prevent --embeddings from poisoning the draft model context.
    # Without this, embedding=true propagates to the draft via params_dft = params_base,
    # causing output_all=true and n_ubatch mismatch → 0% spec decode acceptance.
    # The sed below IS the patch; CI verifies the target line still exists
    # at the pinned rev (test.yml llama-patches-apply).
    sed -i '/auto params_dft = params_base;/a\        params_dft.embedding = false;  // ATLAS: draft never needs embeddings' \
        tools/server/server-context.cpp && \
    cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES="120;121" && \
    cmake --build build --config Release -j$(nproc)

FROM docker.io/nvidia/cuda:12.9.0-runtime-rockylinux9
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/
RUN mkdir -p /models /templates
COPY templates/ /templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# The entrypoint's default listen port (PORT env) is 8080.
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
