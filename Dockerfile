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
ARG INSTALL_CROWD_SHIELD=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Bind on all interfaces inside the container; publish it deliberately.
    HOST=0.0.0.0 \
    PORT=8000 \
    # All mutable state (cameras.json, uploads, samples, alerts.db) in one volume.
    CROWD_DATA_DIR=/app/.data/data \
    CROWD_UPLOADS_DIR=/app/.data/uploads \
    CROWD_SAMPLES_DIR=/app/.data/samples \
    # Third-party caches must not land in $HOME (would be lost with the layer).
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

# Optional dense-crowd backend.  The package owns the PET / DM-Count network
# definitions and retrieves its checkpoints through its documented loader.
# Its current pyproject metadata names a non-existent package and pins Python
# to <3.11.  The runtime package is plain Python and all required dependencies
# are already installed above, so copy only its `crowdshield` source package.
# Keep the default detector-only image small and deterministic; production
# dense nodes opt in with `--build-arg INSTALL_CROWD_SHIELD=1`.
RUN if [ "${INSTALL_CROWD_SHIELD}" = "1" ]; then \
      python -c "import io, pathlib, urllib.request, zipfile; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://github.com/khushi-git-7/Crowd_shield/archive/refs/heads/main.zip').read())); root=pathlib.Path('/usr/local/lib/python3.11/site-packages'); files=[x for x in z.namelist() if x.startswith('Crowd_shield-main/crowdshield/') and not x.endswith('/')]; [(root / x.split('/', 1)[1]).parent.mkdir(parents=True, exist_ok=True) or (root / x.split('/', 1)[1]).write_bytes(z.read(x)) for x in files];"; \
    fi

# Fetch the signed public-release checkpoints while the image is root-owned.
# This prevents first-use downloads from failing under the non-root app user.
RUN if [ "${INSTALL_CROWD_SHIELD}" = "1" ]; then \
      python -c "from crowdshield.points import WEIGHTS, WEIGHTS_SHA256, WEIGHTS_URL; from crowdshield.density import DEFAULT_WEIGHTS, WEIGHTS_SHA256 as DENSITY_SHA256, WEIGHTS_URL as DENSITY_URL; from crowdshield.weights import fetch; fetch(WEIGHTS, WEIGHTS_URL, WEIGHTS_SHA256); fetch(DEFAULT_WEIGHTS, DENSITY_URL, DENSITY_SHA256)"; \
    fi

# Application code (context is filtered by .dockerignore).
COPY app ./app
COPY static ./static
COPY run.py ./
COPY tools ./tools
COPY README.md ./

# Bake the weights in so the container starts with no internet access. This is a
# build-time download (a few MB) rather than a COPY because `*.pt` is gitignored:
# on a fresh clone there is no local file to copy, and `COPY yolo26n.pt*` would
# fail the build outright. Set CROWD_MODEL to change the default weight.
ARG WEIGHTS=yolo26n.pt
# Both runtime defaults are baked in: YOLO26n for CPU, RT-DETR-l for GPU.
# Set CROWD_MODEL at runtime to pin one; other families download on demand.
ARG EXTRA_WEIGHTS=rtdetr-l.pt
RUN python -c "from ultralytics import YOLO, RTDETR; YOLO('${WEIGHTS}'); RTDETR('${EXTRA_WEIGHTS}')" \
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
