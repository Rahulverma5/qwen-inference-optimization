"""Rate limiting (token bucket per API key) and admission control (bounded queue).

They answer different questions:
- TokenBucket: is THIS client sending too fast? (fairness between clients)
- Admission: is the BACKEND full? (protects latency for everyone: back-pressure)
Both reject with 429 + Retry-After, so a well-behaved client knows when to come back.
"""

from __future__ import annotations

import asyncio
import math
import time


class TokenBucket:
    """`burst` tokens, refilled continuously at `rate` per second; a request costs one."""

    def __init__(self, rate: float, burst: int, clock=time.monotonic):
        self.rate, self.burst, self._clock = rate, burst, clock
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def try_acquire(self, key: str) -> float:
        """0.0 if allowed, else seconds until a token is available."""
        now = self._clock()
        tokens = min(self.burst, self._tokens.get(key, self.burst)
                     + (now - self._last.get(key, now)) * self.rate)
        self._last[key] = now
        if tokens >= 1:
            self._tokens[key] = tokens - 1
            return 0.0
        self._tokens[key] = tokens
        return (1 - tokens) / self.rate


class Rejected(Exception):
    def __init__(self, reason: str, retry_after_s: float):
        super().__init__(reason)
        self.reason, self.retry_after_s = reason, retry_after_s

    @property
    def retry_after_header(self) -> str:
        return str(max(1, math.ceil(self.retry_after_s)))


class Admission:
    """At most `limit` requests at the backend, at most `max_queue` waiting, each for at most
    `max_wait_s`. A full queue is rejected at once (`queue_full`); a wait that runs out is
    rejected then (`queue_timeout`): by then the request would miss its latency SLO anyway."""

    def __init__(self, limit: int, max_queue: int, max_wait_s: float):
        self.limit, self.max_queue, self.max_wait_s = limit, max_queue, max_wait_s
        self._sem = asyncio.Semaphore(limit)
        self.in_flight = 0
        self.waiting = 0

    async def acquire(self) -> None:
        if self.in_flight >= self.limit and self.waiting >= self.max_queue:
            raise Rejected("queue_full", self.max_wait_s)
        self.waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), self.max_wait_s)
        except asyncio.TimeoutError:  # = TimeoutError on 3.11+
            raise Rejected("queue_timeout", self.max_wait_s) from None
        finally:
            self.waiting -= 1
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1
        self._sem.release()
