#!/usr/bin/env bash
# Start the dashboard. Everything (venv, uploads, alert DB, samples, caches)
# lives inside the project folder on D: -- nothing is written to C:.
# Override with GODAVARI_VENV / GODAVARI_DATA if you ever need to.
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

VENV="${GODAVARI_VENV:-$ROOT/.venv}"
DATA_ROOT="${GODAVARI_DATA:-$ROOT/.data}"

if [ ! -f "$VENV/Scripts/python.exe" ] && [ ! -f "$VENV/bin/python" ]; then
  echo "venv missing at $VENV -- running tools/install.sh"
  bash tools/install.sh || exit 1
fi

export CROWD_DATA_DIR="$DATA_ROOT/data"
export CROWD_UPLOADS_DIR="$DATA_ROOT/uploads"
export CROWD_SAMPLES_DIR="$DATA_ROOT/samples"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"

# Third-party caches (ultralytics settings, torch hub, matplotlib, temp files)
# also stay on the project drive.
export YOLO_CONFIG_DIR="$DATA_ROOT/ultralytics"
export TORCH_HOME="$DATA_ROOT/torch"
export MPLCONFIGDIR="$DATA_ROOT/matplotlib"
export TMPDIR="$ROOT/.tmp"
export TEMP="$ROOT\\.tmp"
export TMP="$ROOT\\.tmp"

mkdir -p "$CROWD_DATA_DIR" "$CROWD_UPLOADS_DIR" "$CROWD_SAMPLES_DIR" \
         "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" "$ROOT/.tmp"

if [ -x "$VENV/bin/python" ]; then
  exec "$VENV/bin/python" run.py
fi
exec "$VENV/Scripts/python.exe" run.py
