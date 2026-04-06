FROM docker.io/python:3.11-slim

WORKDIR /app

# Install system dependencies and language runtimes
RUN apt-get update && apt-get install -y --no-install-recommends \
    # C/C++
    gcc g++ make \
    # Bash/Shell
    bash \
    # Node.js (via nodesource)
    curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npm --version

# Install Go 1.22
RUN curl -fsSL https://go.dev/dl/go1.22.10.linux-amd64.tar.gz | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"
RUN go version

# Install Rust (rustup, stable toolchain)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && cargo --version

# Install TypeScript globally
RUN npm install -g typescript tsx
RUN tsc --version

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pylint \
    pytest \
    pydantic

# Create sandbox workspace
RUN mkdir -p /tmp/sandbox && chmod 777 /tmp/sandbox

COPY executor_server.py .

EXPOSE 8020

CMD ["python", "executor_server.py"]
