"""Redaction helpers for camera-source URLs before remote archival or logs."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_SOURCE_CREDENTIALS = re.compile(
    r"(?i)\b((?:rtsp|rtsps|http|https)://)[^/@\s]+@"
)

def redact_source_credentials(value):
    """Recursively replace embedded auth in supported stream URLs."""
    if isinstance(value, str):
        return _SOURCE_CREDENTIALS.sub(r"\1[redacted]@", value)
    if isinstance(value, dict):
        return {key: redact_source_credentials(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_source_credentials(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_source_credentials(item) for item in value)
    return value

def source_display_name(source: str) -> str:
    """Create a useful source label without persisting auth, path, or query."""
    try:
        parsed = urlsplit(source)
        host = parsed.hostname
        if host:
            return f"{parsed.scheme.upper()} camera · {host}"
    except ValueError:
        pass
    return "Network camera"
