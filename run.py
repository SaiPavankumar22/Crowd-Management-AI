"""Start the dashboard server:  python run.py   (or set HOST/PORT env vars)"""
import json
import logging
import os
import sys
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv

# Load the repository-root .env for local development. Docker Compose also
# passes these values explicitly to the container.
load_dotenv()


class JsonLogFormatter(logging.Formatter):
    """Small structured formatter with no additional runtime dependency."""
    def format(self, record: logging.LogRecord) -> str:
        row = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            row["exception"] = self.formatException(record.exc_info)
        return json.dumps(row, ensure_ascii=False)


def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

if __name__ == "__main__":
    json_logs = os.environ.get("CROWD_LOG_JSON", "0").lower() in ("1", "true", "yes", "on")
    if json_logs:
        configure_logging()
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        # uvicorn reads a str as a config *file*; pass the built-in dict instead,
        # and skip its logging config entirely when our JSON formatter is active.
        log_config=None if json_logs else uvicorn.config.LOGGING_CONFIG,
    )
