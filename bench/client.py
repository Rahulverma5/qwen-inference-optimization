"""Async streaming load generator for any OpenAI-compatible /v1/completions server
(vLLM, llama.cpp server, SGLang, the gateway).

Two arrival modes:
  --concurrency N   closed loop: N requests in flight at all times (capacity curves)
  --rate R          open loop: Poisson arrivals at R req/s (what users actually do; exposes queueing)

    python -m bench.client --url http://localhost:8000 --model qwen \
        --prompts data/workloads/short_v1.jsonl --concurrency 1 2 4 8 16 --out reports/runs/vllm_fp16
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import httpx

from bench.metrics import RequestRecord, summarize
from bench.server_metrics import Sampler, summarize_samples
from tracking import log_run


async def send_one(client: httpx.AsyncClient, url: str, model: str, item: dict, rid: int,
                   ignore_eos: bool) -> RequestRecord:
    body = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": item["max_tokens"],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:  # fixed output length so configs are compared on equal work
        body["ignore_eos"] = True
    rec = RequestRecord(request_id=rid, t_send=time.perf_counter())
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=body) as resp:
            if resp.status_code != 200:
                rec.error = f"HTTP {resp.status_code}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    rec.prompt_tokens = chunk["usage"]["prompt_tokens"]
                    rec.output_tokens = chunk["usage"]["completion_tokens"]
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    rec.chunk_times.append(now)
    except httpx.HTTPError as e:
        rec.error = type(e).__name__
    return rec


async def run_closed_loop(url, model, items, concurrency, ignore_eos, timeout):
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def guarded(i, item):
            async with sem:
                return await send_one(client, url, model, item, i, ignore_eos)

        t0 = time.perf_counter()
        records = await asyncio.gather(*(guarded(i, it) for i, it in enumerate(items)))
        return list(records), time.perf_counter() - t0


async def run_open_loop(url, model, items, rate, ignore_eos, timeout, seed=0):
    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=None)) as c:
        tasks = []
        t0 = time.perf_counter()
        for i, item in enumerate(items):
            tasks.append(asyncio.create_task(send_one(c, url, model, item, i, ignore_eos)))
            await asyncio.sleep(rng.expovariate(rate))
        records = await asyncio.gather(*tasks)
        return list(records), time.perf_counter() - t0


async def measure(url, model, items, level, closed, ignore_eos, timeout, scrape):
    """One sweep level, with the server sampled for the whole window when `scrape` is set."""
    run = (run_closed_loop(url, model, items, level, ignore_eos, timeout) if closed
           else run_open_loop(url, model, items, level, ignore_eos, timeout))
    if not scrape:
        recs, wall = await run
        return recs, wall, None
    async with Sampler(url) as sampler:
        recs, wall = await run
    return recs, wall, sampler


def reset_prefix_cache(url: str) -> bool:
    """Every level replays the same prompts, so without a reset the prefix cache serves later
    levels from earlier ones (Week 2 run 2: short_v1 hit rate 0% -> 87.5%) and TTFT reads low.
    vLLM exposes this endpoint only with VLLM_SERVER_DEV_MODE=1 (bench.server sets it)."""
    try:
        return httpx.post(f"{url}/reset_prefix_cache", timeout=30).status_code == 200
    except httpx.HTTPError:
        return False


def env_metadata(url: str) -> dict:
    meta = {"python": platform.python_version(), "host": platform.node(), "url": url}
    try:
        meta["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        meta["gpu"] = None
    try:
        meta["server_version"] = httpx.get(f"{url}/version", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        meta["server_version"] = None
    return meta


def write_records(path: Path, records: list[RequestRecord]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "ok", "error", "prompt_tokens", "output_tokens",
                    "ttft_s", "e2e_s", "tpot_s", "num_chunks"])
        for r in records:
            w.writerow([r.request_id, r.ok, r.error, r.prompt_tokens, r.output_tokens,
                        r.ttft if r.ok else "", r.e2e if r.ok else "", r.tpot or "",
                        len(r.chunk_times)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--prompts", required=True, help="JSONL from bench.workloads")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--concurrency", type=int, nargs="+")
    mode.add_argument("--rate", type=float, nargs="+")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-ignore-eos", action="store_true")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--label", default="run", help="config label stored with results")
    p.add_argument("--no-scrape", action="store_true",
                   help="don't sample the server's /metrics and nvidia-smi during each level")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    items = [json.loads(line) for line in Path(a.prompts).read_text().splitlines()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ignore_eos = not a.no_ignore_eos
    meta = env_metadata(a.url) | {"label": a.label, "prompts": a.prompts}
    (out / "env.json").write_text(json.dumps(meta, indent=2))

    asyncio.run(run_closed_loop(a.url, a.model, items[: a.warmup], 1, ignore_eos, a.timeout))
    summaries = []
    for level in a.concurrency or a.rate:
        tag = f"c{level}" if a.concurrency else f"r{level}"
        was_reset = reset_prefix_cache(a.url)
        if not was_reset:
            print(f"{tag}: prefix cache NOT reset; TTFT may be optimistic")
        recs, wall, sampler = asyncio.run(measure(
            a.url, a.model, items, level, bool(a.concurrency), ignore_eos, a.timeout,
            scrape=not a.no_scrape))
        write_records(out / f"requests_{tag}.csv", recs)
        s = {"label": a.label, "mode": tag} | summarize(recs, wall)
        s["prefix_cache_reset"] = int(was_reset)
        if sampler:
            sampler.write(out / f"server_{tag}.csv")
            s |= summarize_samples(sampler.rows)
        summaries.append(s)
        print(f"{tag:>6}: {s['request_throughput_rps']:.2f} req/s, "
              f"{s['output_tok_per_s']:.0f} out tok/s, TTFT p95 {s['ttft_s_p95'] * 1e3:.0f} ms, "
              f"TPOT p95 {s['tpot_s_p95'] * 1e3:.1f} ms, errors {s['error_rate']:.1%}, "
              f"KV max {s.get('kv_usage_max', float('nan')):.0%}, "
              f"waiting max {s.get('waiting_max', float('nan')):.0f}")
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    for s in summaries:
        log_run("serving-benchmarks", f"{a.label}-{s['mode']}",
                params={"label": a.label, "mode": s["mode"], "prompts": a.prompts, "url": a.url,
                        "gpu": meta.get("gpu"), "ignore_eos": ignore_eos},
                metrics={k: v for k, v in s.items() if k not in ("label", "mode")},
                artifacts=[out / f"requests_{s['mode']}.csv", out / f"server_{s['mode']}.csv",
                           out / "env.json"])


if __name__ == "__main__":
    main()
