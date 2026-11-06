"""Ensure the backend package root is importable during tests."""
from pathlib import Path
import sys

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
