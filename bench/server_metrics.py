"""Server-side view of a benchmark run: vLLM's Prometheus /metrics plus nvidia-smi.

The client only sees latency. To explain *why* latency bends (KV cache full, requests queued,
preemptions), sample the server while the load runs and keep the timeseries next to the
request CSV.

vLLM metric names changed across versions (V0 `gpu_cache_usage_perc`, V1 `kv_cache_usage_perc`),
so each quantity lists the names it accepts.

Several servers (independent replicas behind one client) are scraped together: queue gauges and
counters are summed across them, KV usage is the fullest server's, since that one preempts first.
"""

from __future__ import annotations

import asyncio
import csv
import re
import time
from pathlib import Path

import httpx

GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
COUNTERS = {
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    # speculative decoding (V1): proposals made, tokens proposed, tokens the target accepted
    "spec_drafts": ("vllm:spec_decode_num_drafts_total",),
    "spec_draft_tokens": ("vllm:spec_decode_num_draft_tokens_total",),
    "spec_accepted": ("vllm:spec_decode_num_accepted_tokens_total",),
}
MAX_ACROSS_SERVERS = {"kv_usage"}

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum each metric over its label sets (one model per server, so labels are just model_name)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not (m := _LINE.match(line)):
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            continue
    return out


def pick(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    return next((metrics[n] for n in names if n in metrics), None)


def combine(per_server: list[dict[str, float | None]]) -> dict[str, float | None]:
    """One row from several servers' readings: sum, except KV usage (max)."""
    out: dict[str, float | None] = {}
    for k in (*GAUGES, *COUNTERS):
        vals = [r[k] for r in per_server if r.get(k) is not None]
        out[k] = (max(vals) if k in MAX_ACROSS_SERVERS else sum(vals)) if vals else None
    return out


async def gpu_mem_used_mib() -> float | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        return max(float(x) for x in out.decode().split())  # busiest visible GPU
    except (OSError, ValueError):
        return None


class Sampler:
    """Polls the server(s) every `interval_s` until stopped. Use as `async with Sampler(...) as s:`."""

    def __init__(self, urls: str | list[str], interval_s: float = 1.0):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.interval_s = interval_s
        self.rows: list[dict] = []
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            t0 = time.perf_counter()
            while True:
                row: dict = {"t_s": time.perf_counter() - t0}
                readings = []
                for url in self.urls:
                    try:
                        m = parse_prometheus((await c.get(f"{url}/metrics")).text)
                        readings.append({k: pick(m, v) for k, v in (GAUGES | COUNTERS).items()})
                    except httpx.HTTPError:
                        pass
                # A missed scrape of one replica would make summed counters dip; skip the row.
                if len(readings) == len(self.urls):
                    row |= combine(readings)
                row["gpu_mem_mib"] = await gpu_mem_used_mib()
                self.rows.append(row)
                await asyncio.sleep(self.interval_s)

    async def __aenter__(self) -> Sampler:
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def write(self, path: Path) -> None:
        if not self.rows:
            return
        fields = ["t_s", *GAUGES, *COUNTERS, "gpu_mem_mib"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)


def summarize_samples(rows: list[dict]) -> dict[str, float]:
    """Peaks of the gauges, and counter deltas over the window (counters are cumulative)."""
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    out: dict[str, float] = {}
    for k in ("running", "waiting", "kv_usage", "gpu_mem_mib"):
        v = col(k)
        out[f"{k}_max"] = max(v) if v else float("nan")
    for k in COUNTERS:
        v = col(k)
        out[f"{k}_delta"] = v[-1] - v[0] if len(v) >= 2 else float("nan")
    q, h = out["prefix_queries_delta"], out["prefix_hits_delta"]
    out["prefix_hit_rate"] = h / q if q and q == q else float("nan")  # q == q: not NaN
    # Acceptance rate = accepted / proposed tokens; mean accepted length counts the bonus token
    # the target always emits, so 1.0 means speculation never helped.
    d, dt, acc = out["spec_drafts_delta"], out["spec_draft_tokens_delta"], out["spec_accepted_delta"]
    out["spec_acceptance_rate"] = acc / dt if dt and dt == dt else float("nan")
    out["spec_mean_accepted_len"] = 1 + acc / d if d and d == d else float("nan")
    for k in ("prefix_queries_delta", "prefix_hits_delta", "spec_drafts_delta",
              "spec_draft_tokens_delta", "spec_accepted_delta"):
        del out[k]
    return out
