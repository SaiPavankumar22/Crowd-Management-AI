"""Video source helpers: uploaded files, bundled samples, RTSP, webcam."""
import os
import threading
import uuid

from fastapi import UploadFile

from . import config

_upload_lock = threading.Lock()


class UploadTooLarge(ValueError):
    pass


def _valid_video_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in config.VIDEO_EXTS


def save_upload(upload: UploadFile) -> str:
    if not upload.filename or not _valid_video_name(upload.filename):
        raise ValueError(f"unsupported file type: {upload.filename!r} "
                         f"(allowed: {', '.join(config.VIDEO_EXTS)})")
    ext = os.path.splitext(upload.filename)[1].lower()
    path = os.path.join(config.UPLOADS_DIR, f"{uuid.uuid4().hex}{ext}")
    written = 0
    try:
        with _upload_lock, open(path, "wb") as fh:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise UploadTooLarge(f"video exceeds maximum size of {config.MAX_UPLOAD_BYTES} bytes")
                fh.write(chunk)
    except Exception:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        raise
    return path


def list_samples() -> list[str]:
    """Bundled clips, including one level of subfolders (e.g. pilgrim/).

    Subfolder clips are returned as relative names so the client can group them.
    """
    if not os.path.isdir(config.SAMPLES_DIR):
        return []
    found = []
    for entry in os.scandir(config.SAMPLES_DIR):
        if entry.is_file() and _valid_video_name(entry.name):
            found.append(entry.name)
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and _valid_video_name(sub.name):
                    found.append(f"{entry.name}/{sub.name}")
    return sorted(found)


def sample_path(name: str) -> str:
    # Reject traversal: the resolved file must stay inside SAMPLES_DIR.
    candidate = os.path.normpath(os.path.join(config.SAMPLES_DIR, name.replace("\\", "/")))
    root = os.path.normpath(config.SAMPLES_DIR)
    if (not _valid_video_name(candidate) or not os.path.isfile(candidate)
            or os.path.commonpath([root, candidate]) != root):
        raise ValueError(f"sample not found: {name}")
    return candidate
