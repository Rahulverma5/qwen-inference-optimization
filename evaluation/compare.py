"""Compare two prediction files on Spider: where did accuracy change, and why.

    python -m evaluation.compare --spider-dir data/spider_data \
        --a reports/week1/run4/preds/base_fp16.jsonl --b reports/week1/run4/preds/sft_fp16.jsonl \
        --out reports/week1/base_vs_sft.json

Reused in Week 3 to compare quantized variants against the fp16 model.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from evaluation.spider import db_path, execute, results_match

# Spider's official hardness buckets need its full SQL parser; this keyword proxy is close
# enough to see where gains come from, and is labelled as a proxy in reports.
_NESTED = re.compile(r"\(\s*select", re.I)
_SETOP = re.compile(r"\b(union|intersect|except)\b", re.I)


def complexity(sql: str) -> str:
    s = sql.lower()
    if _NESTED.search(s) or _SETOP.search(s):
        return "nested/set-op"
    if " join " in s and (" group by " in s or " order by " in s):
        return "join+agg/order"
    if " join " in s:
        return "join"
    if " group by " in s or " order by " in s or " having " in s:
        return "agg/order"
    return "simple"


def classify(spider_dir: Path, ex: dict, pred: str) -> tuple[bool, str | None]:
    """(correct, error kind if the prediction failed to execute)."""
    path = db_path(spider_dir, ex["db_id"])
    try:
        rows = execute(path, pred)
    except sqlite3.Error as e:
        msg = str(e)
        kind = re.sub(r":.*", "", msg) if ":" in msg else msg
        return False, kind[:60]
    gold = execute(path, ex["query"])
    return results_match(rows, gold, ordered="order by" in ex["query"].lower()), None


def load(path: Path) -> dict[int, str]:
    return {r["id"]: r["sql"] for r in map(json.loads, path.read_text().splitlines())}


def compare(spider_dir: Path, examples: list[dict], a: dict[int, str], b: dict[int, str]) -> dict:
    flips = Counter()
    by_bucket: dict[str, Counter] = {}
    errors = {"a": Counter(), "b": Counter()}
    fixed, broken = [], []
    for i, ex in enumerate(examples):
        ok_a, err_a = classify(spider_dir, ex, a.get(i, ""))
        ok_b, err_b = classify(spider_dir, ex, b.get(i, ""))
        flips[(ok_a, ok_b)] += 1
        c = by_bucket.setdefault(complexity(ex["query"]), Counter())
        c["n"] += 1
        c["a"] += ok_a
        c["b"] += ok_b
        for side, err in (("a", err_a), ("b", err_b)):
            if err:
                errors[side][err] += 1
        if not ok_a and ok_b:
            fixed.append(i)
        if ok_a and not ok_b:
            broken.append(i)
    n = len(examples)
    return {
        "n": n,
        "acc_a": sum(v for (ok, _), v in flips.items() if ok) / n,
        "acc_b": sum(v for (_, ok), v in flips.items() if ok) / n,
        "both_right": flips[(True, True)],
        "fixed_by_b": flips[(False, True)],
        "broken_by_b": flips[(True, False)],
        "both_wrong": flips[(False, False)],
        "by_complexity_proxy": {
            k: {"n": c["n"], "acc_a": round(c["a"] / c["n"], 3), "acc_b": round(c["b"] / c["n"], 3)}
            for k, c in sorted(by_bucket.items(), key=lambda kv: -kv[1]["n"])
        },
        "exec_errors_a": dict(errors["a"].most_common(6)),
        "exec_errors_b": dict(errors["b"].most_common(6)),
        "examples_fixed": fixed[:10],
        "examples_broken": broken[:10],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--split", default="dev.json")
    p.add_argument("--a", type=Path, required=True, help="reference predictions")
    p.add_argument("--b", type=Path, required=True, help="candidate predictions")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    examples = json.loads((a.spider_dir / a.split).read_text())
    report = {"a": str(a.a), "b": str(a.b)} | compare(a.spider_dir, examples, load(a.a), load(a.b))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
