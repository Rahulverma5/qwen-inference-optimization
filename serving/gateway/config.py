"""Gateway settings, all from environment variables (12-factor: the same image runs locally, in
compose, on Cloud Run and on EC2; only the env changes).

Defaults follow the Week 3 ship decision: one llama-server slot per instance, so ONE request in
flight, a short bounded queue, and a fast 429 instead of a request that would miss the 10 s p95
SLO anyway (one uncached request takes ~4-9 s on 4 vCPUs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(f"GATEWAY_{name}", default)


@dataclass(frozen=True)
class Settings:
    backend_url: str = "http://localhost:8080"   # llama-server
    model_name: str = "qwen-sql"
    api_keys: tuple[str, ...] = ()
    auth_disabled: bool = False
    # token bucket per API key: `burst` requests at once, refilled at `rate_per_s`
    rate_per_s: float = 1.0
    burst: int = 5
    # admission: requests sent to the backend at once, how many may wait, and for how long
    max_in_flight: int = 1
    max_queue: int = 2
    max_queue_wait_s: float = 6.0
    request_timeout_s: float = 60.0    # total backend time per request -> 504
    connect_timeout_s: float = 2.0
    max_input_chars: int = 8000        # ~2k tokens: the per-slot context (ctx-size 2048) -> 413
    max_tokens_cap: int = 256
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        keys = tuple(k.strip() for k in _env("API_KEYS", "").split(",") if k.strip())
        s = cls(
            backend_url=_env("BACKEND_URL", cls.backend_url).rstrip("/"),
            model_name=_env("MODEL_NAME", cls.model_name),
            api_keys=keys,
            auth_disabled=_env("AUTH_DISABLED", "false").lower() in {"1", "true", "yes"},
            rate_per_s=float(_env("RATE_PER_S", str(cls.rate_per_s))),
            burst=int(_env("BURST", str(cls.burst))),
            max_in_flight=int(_env("MAX_IN_FLIGHT", str(cls.max_in_flight))),
            max_queue=int(_env("MAX_QUEUE", str(cls.max_queue))),
            max_queue_wait_s=float(_env("MAX_QUEUE_WAIT_S", str(cls.max_queue_wait_s))),
            request_timeout_s=float(_env("REQUEST_TIMEOUT_S", str(cls.request_timeout_s))),
            connect_timeout_s=float(_env("CONNECT_TIMEOUT_S", str(cls.connect_timeout_s))),
            max_input_chars=int(_env("MAX_INPUT_CHARS", str(cls.max_input_chars))),
            max_tokens_cap=int(_env("MAX_TOKENS_CAP", str(cls.max_tokens_cap))),
            log_level=_env("LOG_LEVEL", cls.log_level),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not self.api_keys and not self.auth_disabled:
            raise ValueError("set GATEWAY_API_KEYS (comma-separated) or GATEWAY_AUTH_DISABLED=true")
        if self.max_in_flight < 1 or self.max_queue < 0 or self.rate_per_s <= 0 or self.burst < 1:
            raise ValueError("max_in_flight >= 1, max_queue >= 0, rate_per_s > 0, burst >= 1")
