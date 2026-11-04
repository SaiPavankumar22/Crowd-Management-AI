#!/usr/bin/env bash
# Deps installer for the Godavari crowd monitor.
# Creates a project-local venv at ./.venv (override with GODAVARI_VENV=...).
set -u
cd "$(dirname "$0")/.." || exit 1

VENV="${GODAVARI_VENV:-$(pwd)/.venv}"
if [ ! -f "$VENV/Scripts/python.exe" ] && [ ! -f "$VENV/bin/python" ]; then
  python -m venv --system-site-packages "$VENV" || exit 1
fi
# Windows venvs use Scripts/, POSIX venvs use bin/.
if [ -f "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"
else
  PY="$VENV/bin/python"
fi

echo "== python: $PY =="
"$PY" -m pip install --upgrade pip || exit 1
# Single source of truth: pinned versions live in requirements.txt (the same
# file the Docker image installs). torch comes from the CUDA step below.
"$PY" -m pip install -r requirements.txt || exit 1

echo "== attempting CUDA torch (optional; pass SKIP_CUDA=1 to skip) =="
if [ "${SKIP_CUDA:-0}" != "1" ]; then
  # torchvision must come from the same index, or its C++ ops stop matching torch.
  if "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall >>"$VENV/torch_install.log" 2>&1; then
    echo "CUDA_TORCH=OK"
  else
    echo "CUDA_TORCH=FAILED (falling back to the system CPU torch; see $VENV/torch_install.log)"
  fi
fi

"$PY" - <<'EOF'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
import ultralytics, supervision, fastapi, cv2
print("ultralytics:", ultralytics.__version__, "| supervision:", supervision.__version__, "| opencv:", cv2.__version__)
EOF
echo "== install complete =="
