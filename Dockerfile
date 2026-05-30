# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        liblmdb-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --no-install-project --frozen && \
    VENV_PY=.venv/bin/python && \
    uv pip uninstall --python $VENV_PY \
        nvidia-cublas-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 \
        nvidia-curand-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 \
        nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 nvidia-cuda-runtime-cu13 \
        nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 \
        nvidia-cusparselt-cu13 nvidia-cufile-cu13 \
        triton 2>/dev/null || true && \
    uv pip install --python $VENV_PY --force-reinstall --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        torch && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

COPY src/ ./src/
COPY app/ ./app/

EXPOSE 8000

CMD [".venv/bin/uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
