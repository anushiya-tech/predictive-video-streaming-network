"""JSON-lines logging configured for reproducible experiment runs."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class JsonLineFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)

def get_logger(name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    """Return a named logger that writes JSON lines to its own log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    destination = (Path(log_dir) / f"{name.replace('.', '_')}.jsonl").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not any(getattr(handler, "baseFilename", None) == str(destination) for handler in logger.handlers):
        handler = logging.FileHandler(destination, encoding="utf-8")
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
    return logger

