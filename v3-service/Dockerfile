# python:3.11-slim — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so code edits do not invalidate
# the expensive Torch/tree-sitter layers during local image rebuilds.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# GH #39 v1: tree-sitter for /internal/ast_edit. Wheels ship pre-built so
# this doesn't pull in build-essential. Languages match the selector
# translator — extend when adding new languages.
RUN pip install --no-cache-dir \
        tree_sitter \
        tree_sitter_python \
        tree_sitter_html \
        tree_sitter_javascript

# Copy benchmark modules (V3 pipeline depends on them)
COPY benchmark/ /app/benchmark/
COPY v3-service/main.py /app/main.py
# Structural call-graph reasoning (issue #39): tree-sitter-backed call graph.
# Pure stdlib + the tree-sitter grammars installed above; no extra deps.
COPY v3-service/graph/ /app/graph/

EXPOSE 8070

CMD ["python", "main.py"]
