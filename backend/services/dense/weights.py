"""Validate operator-supplied local model weights.

This vendored helper deliberately does not download checkpoints or contact an
external service. Model distribution belongs to the deployment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

def fetch(path: Path, url: str | None, sha256: str) -> Path:
    """Return an existing verified checkpoint; never download it."""
    path = Path(path)
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"{path.name} is corrupt (checksum mismatch)")
        return path
    raise FileNotFoundError(
        f"{path.name} is not installed; place the verified checkpoint in {path.parent}"
    )
