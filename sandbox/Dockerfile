# python:3.11-slim — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6
WORKDIR /app

# Install system dependencies and language runtimes
RUN apt-get update && apt-get install -y --no-install-recommends \
    # C/C++
    gcc g++ make \
    # Bash/Shell
    bash \
    # Node.js (via nodesource)
    curl ca-certificates gnupg unzip \
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

# Install Rust (rustup, stable toolchain) under /opt so the runtime
# user (non-root) can read it. RUSTUP_HOME stays /opt/rustup at runtime
# (read-only toolchain); CARGO_HOME is repointed to the user's tmpfs
# home by compose so registry/cache writes work on the read-only root.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal \
    && chmod -R a+rX /opt/rustup /opt/cargo
ENV PATH="/opt/cargo/bin:${PATH}"
RUN rustc --version && cargo --version

# Install JavaScript/TypeScript verification tools and package managers.
# Project detection may select npm, pnpm, yarn, or bun from lockfiles; the
# sandbox needs all of them available for build-command verification to be
# accurate.
RUN npm install -g typescript tsx pnpm yarn
RUN tsc --version && pnpm --version && yarn --version

# Install Bun for bun.lockb projects (under /opt for non-root reads).
ENV BUN_INSTALL=/opt/bun
RUN curl -fsSL https://bun.sh/install | bash \
    && chmod -R a+rX /opt/bun
ENV PATH="/opt/bun/bin:${PATH}"
RUN bun --version

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
# Python. Keep the API runtime dependencies in one shared file so the
# container and the unit-test environment cannot drift apart.
COPY requirements-runtime.txt /tmp/sandbox-requirements-runtime.txt
COPY requirements-verify.txt /tmp/sandbox-requirements-verify.txt
RUN pip install --no-cache-dir \
    -r /tmp/sandbox-requirements-runtime.txt \
    -r /tmp/sandbox-requirements-verify.txt

# Create sandbox workspace
RUN mkdir -p /tmp/sandbox && chmod 777 /tmp/sandbox

# Non-root runtime user. Compose may override the uid/gid at runtime
# (ATLAS_SANDBOX_UID/GID, written by `atlas init` to match the host
# user so /workspace writes keep the right owner); HOME comes from the
# compose environment because an overridden uid has no passwd entry.
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -d /home/sandbox -s /bin/bash sandbox \
    && mkdir -p /home/sandbox && chown 1000:1000 /home/sandbox

COPY structured_log.py .
COPY private_values.py .
COPY executor_server.py .

EXPOSE 8020

USER sandbox
ENV HOME=/home/sandbox

CMD ["python", "executor_server.py"]
