"""Spider execution accuracy against a running OpenAI-compatible server (llama.cpp, vLLM, the
gateway), instead of offline vLLM. Same prompts, same scorer as evaluation.generate.

    python -m evaluation.remote --url http://localhost:8080 --model qwen-sql \
        --tokenizer models/fp16 --spider-dir data/spider_data --limit 200 \
        --out reports/week3/preds/gguf_q4_k_m_cpu.jsonl

Week 3: the GGUF on llama.cpp CPU, the engine the cloud deployment uses. Week 5: the daily
canary against the deployed endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from evaluation.spider import db_path, schema_ddl, score
from finetune.prompt import build_messages, extract_sql, render


async def complete_all(url: str, model: str, prompts: list[str], concurrency: int,
                       max_tokens: int, timeout: float) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def one(prompt: str) -> dict:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{url}/v1/completions", json={
                        "model": model, "prompt": prompt, "max_tokens": max_tokens,
                        "temperature": 0.0})
                    r.raise_for_status()
                    text = r.json()["choices"][0]["text"]
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    return {"text": "", "error": type(e).__name__, "latency_s": time.perf_counter() - t0}
                return {"text": text, "error": None, "latency_s": time.perf_counter() - t0}

        return await asyncio.gather(*(one(p) for p in prompts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="qwen-sql")
    p.add_argument("--tokenizer", required=True, help="renders the chat template client-side")
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--split", default="dev.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    examples = json.loads((a.spider_dir / a.split).read_text())[: a.limit]
    prompts = [render(tok, build_messages(schema_ddl(db_path(a.spider_dir, ex["db_id"])), ex["question"]),
                      add_generation_prompt=True) for ex in examples]
    t0 = time.perf_counter()
    outs = asyncio.run(complete_all(a.url, a.model, prompts, a.concurrency, 256, a.timeout))
    elapsed = time.perf_counter() - t0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    preds = {}
    with a.out.open("w") as f:
        for i, o in enumerate(outs):
            preds[i] = extract_sql(o["text"])
            f.write(json.dumps({"id": i, "sql": preds[i], "raw": o["text"], "error": o["error"],
                                "latency_s": round(o["latency_s"], 3)}) + "\n")
    lat = sorted(o["latency_s"] for o in outs)
    result = score(a.spider_dir, examples, preds) | {
        "url": a.url, "gen_time_s": elapsed, "concurrency": a.concurrency,
        "errors": sum(o["error"] is not None for o in outs),
        "latency_s_p50": lat[len(lat) // 2], "latency_s_p95": lat[int(0.95 * (len(lat) - 1))]}
    a.out.with_suffix(".score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
