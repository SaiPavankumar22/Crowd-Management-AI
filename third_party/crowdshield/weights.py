"""Download trained model weights from the project's GitHub releases on first use."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/khushi-git-7/Crowd_shield/releases/download"


def fetch(path: Path, url: str, sha256: str) -> Path:
    """Return `path`, downloading it from `url` first if it's missing, and check its SHA-256."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != sha256:
        partial.unlink()
        raise RuntimeError(f"Downloaded {path.name} is corrupt (checksum mismatch)")
    partial.replace(path)
    return path
