# syntax=docker/dockerfile:1
#
# Godavari Crowd Monitor — one image, CPU or GPU.
#
#   CPU (default, no NVIDIA anything):
#     docker build -t godavari:latest .
#
#   GPU (NVIDIA driver + nvidia-container-toolkit on the host):
#     docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#                  -t godavari:gpu .
#     docker run --gpus all -p 8000:8000 -v godavari-data:/app/.data godavari:gpu
#
# The CUDA build installs PyTorch wheels that bundle the CUDA runtime, so the
# base image stays a normal Python image — the host only needs a driver.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm

# Which PyTorch build to install: cpu | cu128 (CUDA 12.8) | cu126 | ...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Bind on all interfaces inside the container; publish it deliberately.
    HOST=0.0.0.0 \
    PORT=8000 \
    # Application packages live under /app/backend.
    PYTHONPATH=/app/backend \
    # All mutable state (cameras.json, uploads, samples, alerts.db) in one volume.
    CROWD_DATA_DIR=/app/.data/data \
    CROWD_UPLOADS_DIR=/app/.data/uploads \
    CROWD_SAMPLES_DIR=/app/.data/samples \
    # Model/cache directories must not land in $HOME (would be lost with the layer).
    YOLO_CONFIG_DIR=/app/.data/ultralytics \
    TORCH_HOME=/app/.data/torch \
    MPLCONFIGDIR=/app/.data/matplotlib \
    XDG_CACHE_HOME=/app/.data/cache

# libgl1/libglib2.0-0: OpenCV's shared-library deps (no X server needed at
# runtime, the wheel only links against them). libgomp1: OpenMP for torch.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch first (from the chosen index) so ultralytics reuses this build instead
# of installing its own CPU wheel on a GPU image — the classic silent-CPU bug.
RUN pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
# One dependency set, the same pins the local venv uses: `opencv-python-headless`
# at these versions ships no cv2 module at all, so the standard wheel plus the
# runtime libs above is the reliable choice in a container.
RUN pip install -r requirements.txt

# Application code: backend/{app,services,tools}, dashboard static assets, runner.
COPY backend ./backend
COPY static ./static
COPY run.py ./
COPY README.md ./
# Local operator-supplied checkpoints (PET / DM-Count / optional detector .pt).
COPY models ./models

# Bake detector defaults so the container starts with no internet access. This is
# a build-time download when the weight is not already present under models/.
ARG WEIGHTS=yolo26n.pt
ARG EXTRA_WEIGHTS=rtdetr-l.pt
RUN python -c "from pathlib import Path; from ultralytics import YOLO, RTDETR; \
Path('models').mkdir(exist_ok=True); \
w=Path('models')/'${WEIGHTS}'; e=Path('models')/'${EXTRA_WEIGHTS}'; \
(YOLO(str(w) if w.exists() else '${WEIGHTS}'), RTDETR(str(e) if e.exists() else '${EXTRA_WEIGHTS}')); \
import shutil; \
[shutil.move(src, f'models/{src}') for src in ('${WEIGHTS}', '${EXTRA_WEIGHTS}') if Path(src).exists()];" \
 && rm -rf /app/ultralytics

# Non-root runtime user. /app/.data is chowned here so a *named* volume inherits
# the ownership; for a bind mount, chown the host folder to 10001:10001.
RUN useradd --uid 10001 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
 && mkdir -p /app/.data/data /app/.data/uploads /app/.data/samples \
             /app/.data/ultralytics /app/.data/torch /app/.data/matplotlib \
             /app/.data/cache \
 && chown -R app:app /app/.data

USER app

EXPOSE 8000

# /api/system answers as soon as the app is up; model load (30-90 s cold) does
# not block it, so start-period covers warm-up without flapping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/system' % os.environ.get('PORT','8000'), timeout=4)"

# Single process on purpose: camera workers, WebSocket clients and the in-memory
# registry are per-process state, so multiple uvicorn workers would split the
# dashboard across them. Scale with cameras per container, not workers.
CMD ["python", "run.py"]
