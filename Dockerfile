FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# /input and /output are provided by the harness at runtime (mounted volumes).
# We still create them so local testing works without extra setup.
RUN mkdir -p /input /output

CMD ["python", "main.py"]
