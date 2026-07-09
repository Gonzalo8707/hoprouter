FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch build: avoids pulling in ~2GB of unused NVIDIA/CUDA
# libraries, since the local model always runs on CPU in this design.
# This keeps the image well under the 10GB submission cap.
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Pre-download the local model weights at BUILD time. This avoids depending
# on network access to Hugging Face during evaluation (the standardized
# scoring environment's egress rules are unknown), and means the model is
# already cached on disk, so the only startup cost at runtime is loading
# it into memory, not downloading it.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# /input and /output are provided by the harness at runtime (mounted volumes).
# We still create them so local testing works without extra setup.
RUN mkdir -p /input /output

CMD ["python", "main.py"]