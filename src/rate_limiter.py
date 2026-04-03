"""
Product-level rate limiter (not per-user).
Tracks requests against a daily cap and a per-minute burst limit.

Alert thresholds (% of daily cap): 40%, 60%, 80%, 100%
All threshold crossings are logged once per threshold per day.
"""

import time
import threading
from collections import deque
from datetime import date
from logger import get_logger

log = get_logger()

# ── Config ──────────────────────────────────────────────────────────────────
DAILY_LIMIT      = 200    # max queries per calendar day
PER_MINUTE_LIMIT = 20     # max queries per rolling 60-second window
ALERT_THRESHOLDS = [0.40, 0.60, 0.80, 1.00]   # fraction of DAILY_LIMIT
# ────────────────────────────────────────────────────────────────────────────


class RateLimitExceeded(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class ProductRateLimiter:
    """
    Thread-safe product-level rate limiter.

    - daily_count   : resets at midnight (UTC day boundary)
    - minute_window : sliding 60-second deque of request timestamps
    - alerted       : set of already-fired threshold fractions (reset daily)
    """

    def __init__(
        self,
        daily_limit: int = DAILY_LIMIT,
        per_minute_limit: int = PER_MINUTE_LIMIT,
        alert_thresholds: list[float] = ALERT_THRESHOLDS,
    ):
        self.daily_limit = daily_limit
        self.per_minute_limit = per_minute_limit
        self.alert_thresholds = sorted(alert_thresholds)

        self._lock = threading.Lock()
        self._daily_count = 0
        self._current_day = date.today()
        self._minute_window: deque[float] = deque()
        self._alerted: set[float] = set()

        log.info(
            "Rate limiter initialised",
            extra={
                "event": "rate_limiter_init",
                "daily_limit": daily_limit,
                "per_minute_limit": per_minute_limit,
                "alert_thresholds": alert_thresholds,
            },
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _maybe_reset_day(self):
        today = date.today()
        if today != self._current_day:
            log.info(
                f"Day rollover — resetting daily counter "
                f"(was {self._daily_count}/{self.daily_limit})",
                extra={"event": "day_rollover", "previous_count": self._daily_count},
            )
            self._daily_count = 0
            self._current_day = today
            self._alerted.clear()

    def _prune_minute_window(self):
        cutoff = time.monotonic() - 60.0
        while self._minute_window and self._minute_window[0] < cutoff:
            self._minute_window.popleft()

    def _check_alerts(self):
        fraction = self._daily_count / self.daily_limit
        for threshold in self.alert_thresholds:
            if fraction >= threshold and threshold not in self._alerted:
                self._alerted.add(threshold)
                pct = int(threshold * 100)
                level = logging_level_for(threshold)
                msg = (
                    f"RATE LIMIT ALERT {pct}% — "
                    f"{self._daily_count}/{self.daily_limit} daily queries used"
                )
                getattr(log, level)(
                    msg,
                    extra={
                        "event": "rate_limit_alert",
                        "rl_threshold_pct": pct,
                        "rl_daily_count": self._daily_count,
                        "rl_daily_limit": self.daily_limit,
                    },
                )

    # ── public API ──────────────────────────────────────────────────────────

    def check_and_record(self):
        """
        Call before processing each request.
        Raises RateLimitExceeded if any limit is breached.
        Records the request on success.

        """
        with self._lock:
            self._maybe_reset_day()
            self._prune_minute_window()

            # Per-minute check
            minute_count = len(self._minute_window)
            if minute_count >= self.per_minute_limit:
                log.warning(
                    f"Per-minute limit hit ({minute_count}/{self.per_minute_limit})",
                    extra={
                        "event": "rate_limit_minute_exceeded",
                        "rl_minute_count": minute_count,
                        "rl_minute_limit": self.per_minute_limit,
                    },
                )
                raise RateLimitExceeded(
                    f"Rate limit: max {self.per_minute_limit} requests/minute. Slow down."
                )

            # Daily check
            if self._daily_count >= self.daily_limit:
                log.error(
                    f"Daily limit hit ({self._daily_count}/{self.daily_limit})",
                    extra={
                        "event": "rate_limit_daily_exceeded",
                        "rl_daily_count": self._daily_count,
                        "rl_daily_limit": self.daily_limit,
                    },
                )
                raise RateLimitExceeded(
                    f"Daily query limit of {self.daily_limit} reached. Resets at midnight."
                )

            # Record
            self._daily_count += 1
            self._minute_window.append(time.monotonic())
            self._check_alerts()

    def status(self) -> dict:
        with self._lock:
            self._maybe_reset_day()
            self._prune_minute_window()
            return {
                "daily_count": self._daily_count,
                "daily_limit": self.daily_limit,
                "daily_remaining": self.daily_limit - self._daily_count,
                "daily_pct_used": round(self._daily_count / self.daily_limit * 100, 1),
                "minute_count": len(self._minute_window),
                "minute_limit": self.per_minute_limit,
            }


def logging_level_for(threshold: float) -> str:
    if threshold >= 1.00:
        return "error"
    if threshold >= 0.80:
        return "warning"
    return "info"


# Module singleton — import this directly
limiter = ProductRateLimiter()
