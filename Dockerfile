FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Pre-download the local model weights at BUILD time. This avoids depending
# on network access to Hugging Face during evaluation (the standardized
# scoring environment's egress rules are unknown), and means the model is
# already cached on disk, so the only startup cost at runtime is loading
# it into memory, not downloading it.
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

# /input and /output are provided by the harness at runtime (mounted volumes).
# We still create them so local testing works without extra setup.
RUN mkdir -p /input /output

CMD ["python", "main.py"]
