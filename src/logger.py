"""
Centralised logger for the Surgery RAG API.
- Writes structured JSON logs to logs/api.log (rotating, 10MB × 5 files)
- Also streams human-readable lines to stdout
- Import get_logger() anywhere to get the same logger instance
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "api.log")
MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5                # keep 5 rotated files


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Any extra fields passed via extra={} are merged in
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "message",
            ):
                continue
            payload[key] = val

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        base = f"{colour}[{ts}] {record.levelname:<8}{self.RESET} {record.getMessage()}"

        extras = {
            k: v for k, v in record.__dict__.items()
            if k.startswith("rl_") or k in (
                "pipeline", "latency_total_s", "question_preview",
                "cost_usd", "event"
            )
        }
        if extras:
            extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extras.items())
            base += extra_str

        return base


def _build_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("surgery_rag")

    if logger.handlers:   # already initialised (e.g. on hot-reload)
        return logger

    logger.setLevel(logging.DEBUG)

    # Rotating JSON file handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    # Human-readable stdout handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(HumanFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# Module-level singleton
log = _build_logger()


def get_logger() -> logging.Logger:
    return log
