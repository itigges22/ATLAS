FROM docker.io/golang:1.24-alpine AS builder

WORKDIR /build
COPY go.mod ./
COPY *.go ./

RUN go build -o atlas-proxy-v2 .

FROM docker.io/alpine:3.20
RUN apk add --no-cache curl bash
COPY --from=builder /build/atlas-proxy-v2 /usr/local/bin/atlas-proxy-v2

EXPOSE 8090
CMD ["atlas-proxy-v2"]
