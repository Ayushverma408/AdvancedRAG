"""
Centralised logger for the Book RAG API.
- Timestamps in IST (UTC+5:30)
- Structured JSON logs → logs/api.log (rotating 10MB × 5 files)
- Human-readable coloured stdout
- Import get_logger() anywhere to get the same instance
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone, timedelta

LOG_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE  = os.path.join(LOG_DIR, "api.log")
MAX_BYTES    = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5

IST = timezone(timedelta(hours=5, minutes=30))

# Fields that are internal to LogRecord — never forwarded as extras
_SKIP_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "message",
    "taskName",
})

# Extra fields we want to surface on stdout (in order)
_HUMAN_FIELDS = [
    "event",
    "book_key",
    "pipeline",
    "question",
    "question_preview",
    "answer_preview",
    "latency_total_s",
    "latency_retrieval_s",
    "latency_llm_s",
    "chunks_returned",
    "pages",
    "cost_usd",
    "request_id",
    "rl_daily_count",
    "rl_daily_remaining",
    "rl_minute_count",
]


class JSONFormatter(logging.Formatter):
    """One JSON object per line — every extra field included."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":    datetime.now(IST).isoformat(),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }

        for key, val in record.__dict__.items():
            if key in _SKIP_KEYS or key.startswith("_"):
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
    RESET  = "\033[0m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        ts     = datetime.now(IST).strftime("%d %b %Y  %I:%M:%S %p IST")

        header = (
            f"{colour}{self.BOLD}[{record.levelname}]{self.RESET}"
            f"{self.DIM}  {ts}{self.RESET}"
            f"  {record.getMessage()}"
        )

        parts = []
        rec_dict = record.__dict__

        for field in _HUMAN_FIELDS:
            val = rec_dict.get(field)
            if val is None:
                continue
            # Pages list → compact string
            if field == "pages" and isinstance(val, list):
                val = ", ".join(f"p.{p}" for p in val)
            # Costs → nicely formatted
            if field == "cost_usd":
                val = f"${val:.6f}"
            parts.append(f"  {self.DIM}{field}{self.RESET}={self.BOLD}{val}{self.RESET}")

        if record.exc_info:
            parts.append(f"\n{self.COLOURS['ERROR']}{self.formatException(record.exc_info)}{self.RESET}")

        return header + "".join(parts)


def _build_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("book_rag")

    if logger.handlers:   # already initialised (hot-reload guard)
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Rotating JSON file (DEBUG+) ──────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())

    # ── Coloured stdout (INFO+) ──────────────────────────────────────────
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(HumanFormatter())

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _build_logger()


def get_logger() -> logging.Logger:
    return log
