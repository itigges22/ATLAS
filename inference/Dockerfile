FROM docker.io/nvidia/cuda:12.9.0-devel-rockylinux9 AS builder
RUN dnf install -y git cmake gcc-c++ && dnf clean all
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp && \
    cd /llama.cpp && \
    # ATLAS patch: prevent --embeddings from poisoning the draft model context.
    # Without this, embedding=true propagates to the draft via params_dft = params_base,
    # causing output_all=true and n_ubatch mismatch → 0% spec decode acceptance.
    # See: patches/fix-embeddings-spec-decode.patch for details.
    sed -i '/auto params_dft = params_base;/a\        params_dft.embedding = false;  // ATLAS: draft never needs embeddings' \
        tools/server/server-context.cpp && \
    cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=120;121 && \
    cmake --build build --config Release -j$(nproc)

FROM docker.io/nvidia/cuda:12.9.0-runtime-rockylinux9
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/
RUN mkdir -p /models /templates
COPY templates/ /templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
