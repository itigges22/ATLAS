FROM docker.io/golang:1.24-alpine AS builder

WORKDIR /build
COPY go.mod ./
COPY *.go ./

RUN go build -o atlas-proxy-v2 .

FROM docker.io/alpine:3.20

# Runtime toolchain for the model's run_command tool. Without these the proxy
# was alpine-bare (curl + bash only), so any `python -m py_compile`, `pytest`,
# `node`, `gcc`, or `make` invocation from the agent loop hit "command not
# found" — model interpreted the microsecond exit times as "Python isn't
# installed" and bailed. See ISSUES.md PC-032.
#
# The proper architectural fix is to route run_command through the sandbox
# container's /execute endpoint (sandbox already has Python and is properly
# isolated). Until that lands, baking the toolchain in here gets the verify
# loop unblocked.
RUN apk add --no-cache \
        curl bash \
        python3 py3-pip \
        nodejs npm \
        gcc g++ musl-dev make \
        git

COPY --from=builder /build/atlas-proxy-v2 /usr/local/bin/atlas-proxy-v2

EXPOSE 8090
CMD ["atlas-proxy-v2"]
