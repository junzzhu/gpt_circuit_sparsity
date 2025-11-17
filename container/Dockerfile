FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir \
    "git+https://github.com/vllm-project/llm-compressor.git@0.8.1" \
    "transformers>=4.40" \
    "accelerate" \
    "safetensors"

# Clone and install gpt_circuit_sparsity in editable mode (source lives under /app)
RUN git clone https://github.com/junzzhu/gpt_circuit_sparsity.git /app/gpt_circuit_sparsity
RUN pip install -e /app/gpt_circuit_sparsity

RUN git clone https://github.com/junzzhu/sparsegpt.git /app/sparsegpt

# Copy your recipe and Python script into the container
# COPY recipe.yaml .
# COPY quantize_llama3.py .

# Set the default command to run when the container starts
# CMD ["python3", "quantize_llama3.py"]
CMD ["sh", "-c", "sleep infinity"]
