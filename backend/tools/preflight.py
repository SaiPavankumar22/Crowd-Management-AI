"""Check the model weights in models/ before the app tries to load them.

The detector, PET and DM-Count checkpoints are stored with Git LFS: a clone
that never ran ``git lfs pull`` has 130-byte pointer files instead of weights.
PyTorch only fails later, deep inside model construction, with a message that
does not mention LFS at all.  This check turns that into one actionable line.

    python tools/preflight.py     # exit 0 when the app can start
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from app import config  # noqa: E402

LFS_MAGIC = b"version https://git-lfs.github.com/spec"
WEIGHT_SUFFIXES = {".pt", ".pth"}
# (env var that enables it, config attribute, checkpoint file name)
ESTIMATORS = (
    ("CROWD_POINT_BACKEND", "POINT_BACKEND", "pet_shab.pth"),
    ("CROWD_DENSITY_BACKEND", "DENSITY_BACKEND", "dmcount_shab.pth"),
)
LFS_FIX = "git lfs install && git lfs pull"


def is_lfs_pointer(path: Path) -> bool:
    """True for an unresolved Git LFS pointer, never for a real checkpoint."""
    try:
        if path.stat().st_size > 2048:
            return False
        with path.open("rb") as handle:
            return handle.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def check() -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors mean the app cannot run correctly."""
    errors: list[str] = []
    warnings: list[str] = []
    models = Path(config.MODELS_DIR)

    pointers = [p for p in sorted(models.glob("*"))
                if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
                and is_lfs_pointer(p)]
    if pointers:
        names = ", ".join(p.name for p in pointers)
        errors.append(
            f"{names} in {models} are Git LFS pointer files, not weights. "
            f"Fetch them with:  {LFS_FIX}")

    # ultralytics can download the detector, so a missing file only warns.
    detector = models / config.MODEL_NAME
    if not detector.exists() and not is_lfs_pointer(detector):
        warnings.append(
            f"detector {config.MODEL_NAME} is not in {models}; ultralytics will "
            "download it on first load (needs internet)")

    for env_name, attr, filename in ESTIMATORS:
        if getattr(config, attr) != "dense":
            continue
        if not (models / filename).exists():
            errors.append(
                f"{env_name}=dense but {filename} is missing from {models}. "
                f"Add the checkpoint ({LFS_FIX}) or leave {env_name} empty.")

    return errors, warnings


def main(verbose: bool = False) -> int:
    errors, warnings = check()
    for line in warnings:
        print(f"warning: {line}", file=sys.stderr)
    for line in errors:
        print(f"error: {line}", file=sys.stderr)
    if errors:
        print("Model weights are not usable on this machine.", file=sys.stderr)
        return 1
    if verbose:
        print(f"weights ok: {config.MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(verbose=True))
