"""Spider execution accuracy.

Expected layout (the official Spider release; also mirrored as Kaggle datasets):
  <spider_dir>/dev.json, train_spider.json, database/<db_id>/<db_id>.sqlite

A prediction counts as correct when it executes and returns the same rows as the gold query.
Rows are compared as multisets unless the gold query has ORDER BY (then order matters).
This is the standard "execution accuracy" simplification; it can over-credit queries that
match by coincidence on the dev DB contents, which is acceptable for relative comparisons.

    python -m evaluation.spider --spider-dir data/spider --preds reports/preds_fp16.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

QUERY_TIMEOUT_S = 5.0


def db_path(spider_dir: Path, db_id: str) -> Path:
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=256)
def schema_ddl(path: Path) -> str:
    """CREATE TABLE statements straight from sqlite_master: the schema the model sees."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    return "\n".join(r[0].strip() + ";" for r in rows)


def execute(path: Path, sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> list[tuple]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    deadline = time.monotonic() + timeout_s
    # Abort runaway queries (cartesian products are common model mistakes).
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    return pred == gold if ordered else Counter(pred) == Counter(gold)


def exec_match(path: Path, pred_sql: str, gold_sql: str) -> tuple[bool, bool]:
    """Returns (pred_executed, correct)."""
    try:
        pred = execute(path, pred_sql)
    except (sqlite3.Error, sqlite3.Warning, ValueError):
        return False, False
    gold = execute(path, gold_sql)
    return True, results_match(pred, gold, ordered="order by" in gold_sql.lower())


def score(spider_dir: Path, examples: list[dict], preds: dict[int, str]) -> dict[str, float]:
    executed = correct = 0
    for i, ex in enumerate(examples):
        ok_exec, ok = exec_match(db_path(spider_dir, ex["db_id"]), preds.get(i, ""), ex["query"])
        executed += ok_exec
        correct += ok
    n = len(examples)
    return {"n": n, "exec_accuracy": correct / n, "valid_sql_rate": executed / n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--split", default="dev.json")
    p.add_argument("--preds", type=Path, required=True,
                   help='JSONL with {"id": <index into split>, "sql": ...}')
    a = p.parse_args()
    examples = json.loads((a.spider_dir / a.split).read_text())
    preds = {}
    for line in a.preds.read_text().splitlines():
        row = json.loads(line)
        preds[row["id"]] = row["sql"]
    print(json.dumps(score(a.spider_dir, examples, preds), indent=2))


if __name__ == "__main__":
    main()
