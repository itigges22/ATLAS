FROM docker.io/python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies for CPU-based LoRA training
RUN pip install --no-cache-dir redis requests transformers peft datasets accelerate

# Install CPU-only PyTorch separately
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY training/ .

RUN chmod +x *.sh *.py

CMD ["./nightly_train.sh"]
