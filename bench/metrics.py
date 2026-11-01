"""Request-level latency metrics and run-level aggregation.

Definitions (all times from time.perf_counter on the client):
  TTFT  = t_first_chunk - t_send            (includes queueing + prefill + network)
  ITL   = gaps between consecutive streamed chunks of one request
  TPOT  = (t_last - t_first) / (output_tokens - 1)   time per output token after the first
  E2E   = t_last - t_send

A streamed chunk can carry more than one token (server-side batching of SSE events), so ITL
is per *chunk* and TPOT uses the server-reported token count. Report both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PERCENTILES = (50, 90, 95, 99)


@dataclass
class RequestRecord:
    request_id: int
    t_send: float
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.chunk_times)

    @property
    def ttft(self) -> float:
        return self.chunk_times[0] - self.t_send

    @property
    def e2e(self) -> float:
        return self.chunk_times[-1] - self.t_send

    @property
    def itls(self) -> list[float]:
        return np.diff(self.chunk_times).tolist()

    @property
    def tpot(self) -> float | None:
        if self.output_tokens < 2:
            return None
        return (self.chunk_times[-1] - self.chunk_times[0]) / (self.output_tokens - 1)


def _pcts(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_p{p}": float("nan") for p in PERCENTILES} | {f"{prefix}_mean": float("nan")}
    arr = np.asarray(values)
    out = {f"{prefix}_p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}
    out[f"{prefix}_mean"] = float(arr.mean())
    return out


def summarize(records: list[RequestRecord], wall_time_s: float) -> dict[str, float]:
    ok = [r for r in records if r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    summary: dict[str, float] = {
        "num_requests": len(records),
        "num_ok": len(ok),
        "error_rate": 1 - len(ok) / len(records) if records else float("nan"),
        "wall_time_s": wall_time_s,
        "request_throughput_rps": len(ok) / wall_time_s,
        "output_tok_per_s": out_tokens / wall_time_s,
        "total_tok_per_s": (out_tokens + sum(r.prompt_tokens for r in ok)) / wall_time_s,
    }
    summary |= _pcts([r.ttft for r in ok], "ttft_s")
    summary |= _pcts([x for r in ok for x in r.itls], "itl_s")
    summary |= _pcts([t for r in ok if (t := r.tpot) is not None], "tpot_s")
    summary |= _pcts([r.e2e for r in ok], "e2e_s")
    return summary
