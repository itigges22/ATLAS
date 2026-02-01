FROM nvidia/cuda:12.8.0-devel-rockylinux9 AS builder
RUN dnf install -y git cmake gcc-c++ && dnf clean all
RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp && \
    cd /llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 && \
    cmake --build build --config Release -j$(nproc)

FROM nvidia/cuda:12.8.0-runtime-rockylinux9
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/
RUN mkdir -p /models /templates
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
