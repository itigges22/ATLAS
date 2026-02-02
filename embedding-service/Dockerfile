FROM docker.io/python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (much smaller than full torch)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Use newer sentence-transformers that is compatible with recent huggingface_hub
RUN pip install --no-cache-dir \
    sentence-transformers>=2.6.0 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.3

# Download the model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY server.py /app/server.py

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
