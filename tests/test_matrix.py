import csv
import json
import math

import pytest

from bench.matrix import (
    load_experiments,
    load_quality,
    md_table,
    pareto_front,
    serving_rows,
    usd_per_million_tokens,
    variant_rows,
)


def test_cost_per_million_tokens():
    # 1,000 tok/s on one $0.36/h GPU: 3.6M tokens/h -> $0.10 per 1M
    assert usd_per_million_tokens(1000, 1, 0.36) == pytest.approx(0.10)
    assert usd_per_million_tokens(1000, 2, 0.36) == pytest.approx(0.20)
    assert math.isnan(usd_per_million_tokens(float("nan"), 1, 0.36))


def test_pareto_front():
    pts = [{"n": "a", "x": 1.0, "y": 0.60}, {"n": "b", "x": 2.0, "y": 0.65},
           {"n": "c", "x": 2.5, "y": 0.62},  # dominated by b
           {"n": "d", "x": 0.5, "y": 0.50}, {"n": "e", "x": float("nan"), "y": 0.9}]
    assert [p["n"] for p in pareto_front(pts, "x", "y")] == ["d", "a", "b"]


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_variant_rows_deltas(tmp_path):
    _write(tmp_path / "preds/fp16.score.json",
           {"n": 100, "exec_accuracy": 0.64, "valid_sql_rate": 0.87, "model_size_gib": 3.2})
    _write(tmp_path / "preds/awq.score.json",
           {"n": 100, "exec_accuracy": 0.62, "valid_sql_rate": 0.86, "model_size_gib": 1.6})
    _write(tmp_path / "ppl/fp16.json", {"wikitext": {"ppl": 10.0}, "spider": {"ppl": 1.2}})
    _write(tmp_path / "ppl/awq.json", {"wikitext": {"ppl": 11.0}})
    _write(tmp_path / "compare/awq.json", {"mcnemar_p": 0.03})
    rows = {r["variant"]: r for r in variant_rows(load_quality([tmp_path]))}
    awq = rows["awq"]
    assert awq["delta_pts"] == pytest.approx(-2.0) and awq["compression_x"] == pytest.approx(2.0)
    assert awq["ppl_wikitext_delta_pct"] == pytest.approx(10.0) and math.isnan(awq["ppl_spider"])
    assert awq["mcnemar_p"] == 0.03 and math.isnan(rows["fp16"]["mcnemar_p"])
    assert "| awq | 100 | 62.0% | -2.0 |" in md_table([awq], [
        ("variant", "V", ""), ("n", "n", ""), ("exec_accuracy", "acc", ".1%"), ("delta_pts", "d", "+.1f")])


def _summary(path, label, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["label", "mode", "output_tok_per_s", "request_throughput_rps", "error_rate",
            "ttft_s_p95", "tpot_s_p95", "e2e_s_p95", "itl_s_p99", "e2e_s_mean"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for mode, tok, ttft in rows:
            w.writerow({"label": label, "mode": mode, "output_tok_per_s": tok,
                        "request_throughput_rps": tok / 30, "error_rate": 0, "ttft_s_p95": ttft,
                        "tpot_s_p95": 0.03, "e2e_s_p95": 1.0, "itl_s_p99": 0.05, "e2e_s_mean": 0.8})


def test_serving_rows_slo_capacity_and_cost(tmp_path):
    for name, gpus, pts in (("fp16", 1, [("c1", 60, 0.1), ("c16", 400, 0.5), ("c64", 600, 2.0)]),
                            ("replicas2", 2, [("c1", 60, 0.1), ("c16", 700, 0.3), ("c64", 1100, 0.6)])):
        _write(tmp_path / name / "result.json",
               {"name": name, "status": "ok", "model": "fp16", "set": {}, "num_gpus": gpus,
                "quality": "fp16", "servers": [{"kv_cache_tokens": 86400, "ready_s": 60}] * gpus})
        _summary(tmp_path / name / "spider_v1/summary.csv", f"{name}/spider_v1", pts)
    results, rows = load_experiments(tmp_path)
    quality = {"fp16": {"score": {"exec_accuracy": 0.646}}}
    slo = {"spider_v1": {"ttft_s_p95": 1.0}}
    t = {r["experiment"]: r for r in serving_rows(results, rows, quality, slo,
                                                  {"workload": "spider_v1", "usd_per_gpu_hour": 0.36})}
    assert t["fp16"]["slo_max_c"] == 16 and t["replicas2"]["slo_max_c"] == 64
    assert t["fp16"]["usd_per_1m_tokens"] == pytest.approx(0.36 / (400 * 3600) * 1e6)
    assert t["fp16"]["usd_per_1k_requests"] == pytest.approx(0.36 / (400 / 30 * 3600) * 1e3)
    assert t["replicas2"]["kv_cache_tokens"] == 2 * 86400 and t["replicas2"]["exec_accuracy"] == 0.646
