# ---- Stage 1: build llama-cpp-python (CPU) ---------------------------------
# llama-cpp-python compiles llama.cpp from source; the build toolchain
# (gcc, cmake) stays in this stage so the final image ships only the wheel.
# Plain CPU build (no BLAS): llama.cpp's own kernels are what we want on
# the scoring host's 2 vCPUs, and it avoids an OpenBLAS runtime dependency.
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_ARGS="-DGGML_BLAS=OFF"
RUN pip wheel --no-cache-dir llama-cpp-python==0.3.2 -w /wheels

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the local model at BUILD time (no network dependency during
# evaluation). Qwen2.5-3B-Instruct Q4_K_M GGUF: 1.93GB single file. Model
# size is dictated by the scoring environment (4GB RAM, 2 vCPU, no GPU):
# 3B Q4 peaks at ~2.4GB resident, leaving headroom; a 7B would be
# OOM-killed. huggingface_hub is only needed for this download.
RUN pip install --no-cache-dir huggingface_hub \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('bartowski/Qwen2.5-3B-Instruct-GGUF', \
'Qwen2.5-3B-Instruct-Q4_K_M.gguf', local_dir='/models')" \
    && pip uninstall -y huggingface_hub

# llama-cpp-python's compiled extension links against libgomp (OpenMP),
# which python:slim does NOT ship - without it `import llama_cpp` fails
# and every local task silently escalates to Fireworks. Installed AFTER
# the model-download layer on purpose: putting it earlier would
# invalidate the cached 1.93GB download on every rebuild.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

ENV LOCAL_MODEL_PATH=/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf

COPY app/ .

# /input and /output are provided by the harness at runtime (mounted volumes).
# We still create them so local testing works without extra setup.
RUN mkdir -p /input /output

CMD ["python", "main.py"]
