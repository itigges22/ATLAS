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
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL https://go.dev/dl/go1.22.10.linux-${ARCH}.tar.gz | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"
RUN go version

# Install Rust (rustup, stable toolchain)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && cargo --version

# Install TypeScript globally
RUN npm install -g typescript tsx
RUN tsc --version

# Pre-baked Python deps — kept INTENTIONALLY MINIMAL after PC-191.
#
# Two groups only:
#   1. executor_server runtime (fastapi/uvicorn/pydantic) — the
#      sandbox API itself depends on these.
#   2. Universal verify helpers (pytest/requests/ruff/mypy) — these
#      are language-agnostic enough that almost every Python project
#      uses one of them, and they're slow to install on demand.
#
# Application libraries (flask/django/numpy/pandas/...) are NOT
# pre-baked. PC-190 originally baked them in to work around the
# read-only fs issue, but that was the wrong layer — PC-191's
# universal tmpfs mounts (see docker-compose.yml) make `pip install`
# work fine, and the project's own venv / requirements.txt is the
# right source of truth for application deps anyway. Keeping the
# image lean means: smaller pull, smaller attack surface, no drift
# between sandbox-baked-flask and user-pinned-flask, and the same
# pattern works for Node/Rust/Go/Ruby/Java/PHP without favoring
# Python.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pytest \
    requests \
    ruff \
    mypy

# Create sandbox workspace
RUN mkdir -p /tmp/sandbox && chmod 777 /tmp/sandbox

COPY executor_server.py .

EXPOSE 8020

CMD ["python", "executor_server.py"]
