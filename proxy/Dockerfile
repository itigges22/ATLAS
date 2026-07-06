# golang:1.24-alpine — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/golang:1.26-alpine@sha256:3ad57304ad93bbec8548a0437ad9e06a455660655d9af011d58b993f6f615648 AS builder
WORKDIR /build
COPY go.mod ./
COPY *.go ./

RUN go build -o atlas-proxy-v2 .

# alpine:3.20 — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/alpine:3.24@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b
# Runtime packages, version-pinned for reproducible builds. Only two are
# needed:
#   curl — the compose/K8s healthcheck (curl -sf localhost:8090/health)
#   bash — runLocally's `bash -c` for ATLAS_VERIFY_IN=host verification
# The proxy binary invokes nothing else directly (its single exec site is
# `bash -c`, tools.go). The language toolchain (python/node/gcc/make/git)
# that used to live here was a pre-PC-188 stopgap: run_command now routes
# to the sandbox container's /shell endpoint, which owns the language
# matrix. Removing it slims the image and shrinks the attack surface
# without changing the verification workflow — sandbox routing is
# unchanged and there is no local execution fallback in the default mode.
#
# NOTE: alpine version pins track the live v3.20 repo, which garbage-
# collects superseded -rN builds; a pin may need bumping when Alpine ships
# a patch (the digest-pinned base keeps the rest of the toolchain fixed).
RUN apk add --no-cache \
        curl=8.21.0-r0 \
        bash=5.3.9-r1

# Non-root application account created inside this image. The proxy reads
# config from env/mounts and writes nothing to the container fs at runtime
# (workspace + secrets are mounts; logs go to stdout/stderr), so no
# writable app directory is required. Running as a named unprivileged user
# drops ambient root in the container.
RUN addgroup -g 1001 -S atlas && adduser -u 1001 -S atlas -G atlas

COPY --from=builder /build/atlas-proxy-v2 /usr/local/bin/atlas-proxy-v2

USER atlas
EXPOSE 8090
CMD ["atlas-proxy-v2"]
