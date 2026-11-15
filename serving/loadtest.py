"""HTTP load test of the gateway's /v1/sql with real Spider dev questions (Week 4).

    python -m serving.loadtest --url http://localhost:8000 --key local-dev-key \
        --spider-dir data/spider_data --n 40 --concurrency 1 --out reports/week4/load/c1.json
    python -m serving.loadtest ... --rate 1.0 --duration 60 --out reports/week4/load/overload.json

Closed loop (--concurrency): N requests always in flight, i.e. capacity. Open loop (--rate):
Poisson arrivals that don't wait for answers, which is what real users do and what exposes
queueing and 429s. Each answer is executed against the Spider DB, so the report also shows that
the served model still writes correct SQL. Questions are drawn in random order (seeded) across
databases, like Week 3's spider_v1, so the prompt cache hits only by chance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from evaluation.spider import db_path, exec_match, schema_ddl


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 3)


async def one(client: httpx.AsyncClient, url: str, key: str, item: dict, t_start: float) -> dict:
    t0 = time.perf_counter()
    rec = {"i": item["i"], "db": item["db_id"], "t_send": round(t0 - t_start, 3)}
    try:
        r = await client.post(f"{url}/v1/sql", headers={"X-API-Key": key},
                              json={"schema": item["schema"], "question": item["question"]})
        rec["status"] = r.status_code
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200:
            rec["sql"] = body["sql"]
            rec["queue_wait_s"] = body["timings"]["queue_wait_s"]
            usage = body.get("usage") or {}
            rec["prompt_cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        else:
            detail = body.get("detail")
            rec["reason"] = detail.get("error") if isinstance(detail, dict) else str(r.status_code)
    except httpx.HTTPError as e:
        rec["status"], rec["reason"] = 0, type(e).__name__
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    return rec


async def run(a, items: list[dict]) -> tuple[list[dict], float]:
    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=a.timeout, limits=httpx.Limits(max_connections=200)) as client:
        if a.rate:
            rng, tasks, t = random.Random(a.seed), [], 0.0
            for item in items:
                t += rng.expovariate(a.rate)
                if t > a.duration:
                    break
                await asyncio.sleep(max(0.0, t_start + t - time.perf_counter()))
                tasks.append(asyncio.create_task(one(client, a.url, a.key, item, t_start)))
            recs = await asyncio.gather(*tasks)
        else:
            sem = asyncio.Semaphore(a.concurrency)

            async def guarded(item):
                async with sem:
                    return await one(client, a.url, a.key, item, t_start)
            recs = await asyncio.gather(*(guarded(it) for it in items))
    return list(recs), time.perf_counter() - t_start


def summarize(recs: list[dict], wall_s: float, spider_dir: Path, gold: dict[int, dict]) -> dict:
    ok = [r for r in recs if r["status"] == 200]
    for r in ok:
        ex = gold[r["i"]]
        r["correct"] = exec_match(db_path(spider_dir, ex["db_id"]), r["sql"], ex["query"])[0]
    by_status: dict[str, int] = {}
    for r in recs:
        k = str(r["status"]) + (f" {r['reason']}" if r.get("reason") else "")
        by_status[k] = by_status.get(k, 0) + 1
    lat_ok = [r["latency_s"] for r in ok]
    lat_rej = [r["latency_s"] for r in recs if r["status"] == 429]
    return {
        "sent": len(recs), "wall_s": round(wall_s, 1), "by_status": by_status,
        "success_rate": round(len(ok) / len(recs), 3) if recs else None,
        "goodput_req_s": round(len(ok) / wall_s, 3),
        "ok_latency_s": {"p50": pct(lat_ok, 0.5), "p95": pct(lat_ok, 0.95), "max": max(lat_ok, default=None)},
        "ok_queue_wait_s_p95": pct([r["queue_wait_s"] for r in ok], 0.95),
        "rejected_latency_s_p95": pct(lat_rej, 0.95),
        "ok_within_10s": round(sum(x <= 10 for x in lat_ok) / len(ok), 3) if ok else None,
        "exec_accuracy_of_answered": round(sum(r["correct"] for r in ok) / len(ok), 3) if ok else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", required=True)
    p.add_argument("--spider-dir", type=Path, default=Path("data/spider_data"))
    p.add_argument("--n", type=int, default=40, help="questions to draw")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rate", type=float, default=None, help="open loop: Poisson arrivals per second")
    p.add_argument("--duration", type=float, default=60.0, help="open loop: seconds of arrivals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    dev = json.loads((a.spider_dir / "dev.json").read_text())
    order = list(range(len(dev)))
    random.Random(a.seed).shuffle(order)
    schemas: dict[str, str] = {}
    items = []
    for i in order[: a.n]:
        ex = dev[i]
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = schema_ddl(db_path(a.spider_dir, ex["db_id"]))
        items.append({"i": i, "db_id": ex["db_id"], "question": ex["question"],
                      "schema": schemas[ex["db_id"]]})

    recs, wall = asyncio.run(run(a, items))
    summary = summarize(recs, wall, a.spider_dir, dict(enumerate(dev))) | {
        "mode": f"open rate={a.rate}/s for {a.duration}s" if a.rate else f"closed c={a.concurrency}",
        "seed": a.seed}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    with a.out.with_suffix(".jsonl").open("w") as f:
        for r in sorted(recs, key=lambda r: r["t_send"]):
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
