"""Audit the Spider data and our prompt format before spending GPU hours.

Answers, with numbers:
  - split sizes, and whether dev databases are unseen in training (cross-database setting)
  - do all gold queries execute; how many return empty results (an empty gold result lets a
    wrong query that also returns nothing count as correct, so it inflates execution accuracy)
  - prompt / completion token lengths under the real Qwen3 chat template -> SFT max_length
  - how much of each prompt is a per-database shared prefix -> expected prefix-cache benefit
  - train prompt == inference prompt, byte for byte

    python -m evaluation.spider_audit --spider-dir data/spider_data --tokenizer Qwen/Qwen3-1.7B \
        --out reports/week1/spider_audit.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.spider import db_path, execute, schema_ddl, score
from finetune.prompt import build_messages, render


def pct(values, ps=(50, 90, 95, 99, 100)) -> dict[str, int]:
    arr = np.asarray(values)
    return {f"p{p}": int(np.percentile(arr, p)) for p in ps} | {"mean": round(float(arr.mean()), 1)}


def gold_execution(spider_dir: Path, examples: list[dict]) -> dict:
    errors, empty, slow = [], 0, 0
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        t = time.perf_counter()
        try:
            rows = execute(db_path(spider_dir, ex["db_id"]), ex["query"])
            empty += not rows
        except sqlite3.Error as e:
            errors.append({"idx": i, "db_id": ex["db_id"], "error": str(e)[:120]})
        slow += time.perf_counter() - t > 1.0
    return {
        "n": len(examples),
        "errors": len(errors),
        "error_examples": errors[:5],
        "empty_result": empty,
        "empty_result_rate": round(empty / len(examples), 4),
        "slow_over_1s": slow,
        "has_order_by": sum("order by" in ex["query"].lower() for ex in examples),
        "total_s": round(time.perf_counter() - t0, 2),
    }


def token_stats(spider_dir: Path, examples: list[dict], tok) -> dict:
    prompt_lens, completion_lens, prefix_lens = [], [], []
    prefix_cache: dict[str, int] = {}
    for ex in examples:
        schema = schema_ddl(db_path(spider_dir, ex["db_id"]))
        prompt = render(tok, build_messages(schema, ex["question"]), add_generation_prompt=True)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        prompt_lens.append(len(p_ids))
        completion_lens.append(len(tok.encode(ex["query"].strip() + tok.eos_token,
                                              add_special_tokens=False)))
        if ex["db_id"] not in prefix_cache:
            # Shared prefix = everything before the question text, identical for one database.
            head = prompt[: prompt.index("Question: ")]
            prefix_cache[ex["db_id"]] = len(tok.encode(head, add_special_tokens=False))
        prefix_lens.append(prefix_cache[ex["db_id"]])
    total = np.add(prompt_lens, completion_lens)
    return {
        "prompt_tokens": pct(prompt_lens),
        "completion_tokens": pct(completion_lens),
        "total_tokens": pct(total),
        "over_1024": int((total > 1024).sum()),
        "over_2048": int((total > 2048).sum()),
        "shared_prefix_fraction_mean": round(float(np.mean(np.divide(prefix_lens, prompt_lens))), 3),
    }


def template_check(spider_dir: Path, ex: dict, tok) -> dict:
    schema = schema_ddl(db_path(spider_dir, ex["db_id"]))
    infer = render(tok, build_messages(schema, ex["question"]), add_generation_prompt=True)
    full = render(tok, build_messages(schema, ex["question"], ex["query"]), add_generation_prompt=False)
    return {
        # SFT trains on prompt + completion; that concatenation must equal the template's own
        # rendering of the full conversation, or the model learns a format it never sees at serve time.
        "train_equals_template": full.startswith(infer)
        and full[len(infer):].strip().startswith(ex["query"].strip()),
        "generation_prompt_tail": infer[-60:],
        "assistant_turn_in_full": full[len(infer) - 30:][:160],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    splits = {s: json.loads((a.spider_dir / f"{s}.json").read_text())
              for s in ("train_spider", "train_others", "dev")}
    train = splits["train_spider"] + splits["train_others"]
    dev = splits["dev"]
    train_dbs = {e["db_id"] for e in train}
    dev_dbs = {e["db_id"] for e in dev}

    report = {
        "sizes": {k: len(v) for k, v in splits.items()},
        "databases": {"train": len(train_dbs), "dev": len(dev_dbs),
                      "dev_seen_in_train": len(dev_dbs & train_dbs)},
        "dev_questions_per_db": pct(list(Counter(e["db_id"] for e in dev).values())),
        "dev_gold_execution": gold_execution(a.spider_dir, dev),
        "dev_gold_vs_gold": score(a.spider_dir, dev, {i: e["query"] for i, e in enumerate(dev)}),
        "tokens_train": token_stats(a.spider_dir, train, tok),
        "tokens_dev": token_stats(a.spider_dir, dev, tok),
        "template": template_check(a.spider_dir, dev[0], tok),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
