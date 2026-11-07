"""Week 3 experiment matrix: speed x quality x memory x cost, and a Pareto plot.

    python -m bench.matrix --experiments reports/week3/experiments \
        --quality reports/week3/quality reports/week3/serving --out reports/week3/report

Inputs
  --experiments  bench.experiments output: <name>/result.json, <name>/<workload>/summary.csv
  --quality      dirs holding preds/<key>.score.json (evaluation.generate / .remote),
                 ppl/<key>.json (evaluation.perplexity), compare/<key>.json
                 (evaluation.compare vs fp16), quant/<key>.json (quantize.compress info)
Outputs
  variants.md     one row per model variant: accuracy, perplexity, size, load time
  serving.md      one row per experiment: SLO capacity and cost on the cost workload,
                  change vs fp16, startup facts, spec-decode acceptance
  detail.csv      every (experiment, workload, concurrency) point, with change vs fp16
  pareto.png      accuracy vs cost per 1M output tokens; the frontier is what's worth shipping
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from bench.report import INK, MUTED, SERIES, SURFACE, _style, closed_loop, load_summaries, max_under_slo

NAN = float("nan")


def usd_per_million_tokens(out_tok_s: float, num_gpus: int, usd_per_gpu_hour: float) -> float:
    """GPU cost of 1M output tokens at a sustained throughput."""
    if not out_tok_s or out_tok_s != out_tok_s:
        return NAN
    return usd_per_gpu_hour * num_gpus / (out_tok_s * 3600) * 1e6


def pareto_front(points: list[dict], x: str, y: str) -> list[dict]:
    """Points not dominated by another with x <= and y >= (one strictly): lower cost, higher
    accuracy. NaN coordinates never make the front. Sorted by x."""
    ok = [p for p in points if p[x] == p[x] and p[y] == p[y]]
    front = [p for p in ok if not any(
        q[x] <= p[x] and q[y] >= p[y] and (q[x] < p[x] or q[y] > p[y]) for q in ok)]
    return sorted(front, key=lambda p: p[x])


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_quality(dirs: list[Path]) -> dict[str, dict]:
    """key -> merged facts from every quality dir (later dirs win on conflicts)."""
    q: dict[str, dict] = {}
    for d in dirs:
        for kind, sub in (("score", "preds"), ("ppl", "ppl"), ("compare", "compare"), ("quant", "quant")):
            pattern = "*.score.json" if kind == "score" else "*.json"
            for f in sorted((d / sub).glob(pattern)):
                key = f.name.removesuffix(".score.json") if kind == "score" else f.stem
                q.setdefault(key, {})[kind] = json.loads(f.read_text())
    return q


def variant_rows(quality: dict[str, dict]) -> list[dict]:
    ref = quality.get("fp16", {})
    ref_acc = ref.get("score", {}).get("exec_accuracy", NAN)
    ref_size = ref.get("score", {}).get("model_size_gib") or NAN
    ref_ppl = {c: ref.get("ppl", {}).get(c, {}).get("ppl", NAN) for c in ("wikitext", "spider")}
    rows = []
    for key, f in quality.items():
        s = f.get("score")
        if not s:
            continue
        size = s.get("model_size_gib") or NAN
        ppl = {c: f.get("ppl", {}).get(c, {}).get("ppl", NAN) for c in ("wikitext", "spider")}
        rows.append({
            "variant": key,
            "n": s["n"],
            "exec_accuracy": s["exec_accuracy"],
            "delta_pts": (s["exec_accuracy"] - ref_acc) * 100 if key != "fp16" else 0.0,
            "mcnemar_p": f.get("compare", {}).get("mcnemar_p", NAN),
            "valid_sql_rate": s["valid_sql_rate"],
            "size_gib": size,
            "compression_x": ref_size / size if size == size else NAN,
            "ppl_wikitext": ppl["wikitext"],
            "ppl_wikitext_delta_pct": (ppl["wikitext"] / ref_ppl["wikitext"] - 1) * 100,
            "ppl_spider": ppl["spider"],
            "ppl_spider_delta_pct": (ppl["spider"] / ref_ppl["spider"] - 1) * 100,
            "quantize_time_s": f.get("quant", {}).get("quantize_time_s", NAN),
            "load_time_s": s.get("load_time_s", NAN),
        })
    return sorted(rows, key=lambda r: (r["variant"] != "fp16", -r["exec_accuracy"]))


def load_experiments(exp_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    results, rows = {}, []
    for rj in sorted(exp_dir.glob("*/result.json")):
        r = json.loads(rj.read_text())
        results[r["name"]] = r
        for summary in sorted(rj.parent.glob("*/summary.csv")):
            rows += [row | {"experiment": r["name"], "workload": summary.parent.name}
                     for row in load_summaries([summary])]
    return results, rows


def detail_rows(rows: list[dict], baseline: str = "fp16") -> list[dict]:
    """Every closed-loop point with its throughput/latency change vs the baseline at equal load."""
    out = []
    by_key = {}
    for r in rows:
        if str(r["mode"]).startswith("c"):
            by_key[(r["experiment"], r["workload"], int(r["mode"][1:]))] = r
    for (exp, wl, c), r in sorted(by_key.items()):
        b = by_key.get((baseline, wl, c))

        def pct(k, b=b, r=r):
            return (r[k] / b[k] - 1) * 100 if b and b[k] else NAN

        out.append({
            "experiment": exp, "workload": wl, "concurrency": c,
            "out_tok_s": r["output_tok_per_s"], "d_tok_s_pct": pct("output_tok_per_s"),
            "ttft_p95_ms": r["ttft_s_p95"] * 1e3, "d_ttft_p95_pct": pct("ttft_s_p95"),
            "tpot_p95_ms": r["tpot_s_p95"] * 1e3, "d_tpot_p95_pct": pct("tpot_s_p95"),
            "itl_p99_ms": r["itl_s_p99"] * 1e3, "e2e_p95_s": r["e2e_s_p95"],
            "error_rate": r["error_rate"], "kv_usage_max": r.get("kv_usage_max", NAN),
            "preemptions": r.get("preemptions_delta", NAN),
            "prefix_hit_rate": r.get("prefix_hit_rate", NAN),
            "spec_acceptance_rate": r.get("spec_acceptance_rate", NAN),
            "spec_mean_accepted_len": r.get("spec_mean_accepted_len", NAN),
        })
    return out


def serving_rows(results: dict[str, dict], rows: list[dict], quality: dict[str, dict],
                 slos: dict, cost: dict) -> list[dict]:
    wl = cost["workload"]
    table = []
    for name, r in results.items():
        sweep = closed_loop([x for x in rows if x["experiment"] == name], f"{name}/{wl}")
        best = max_under_slo(sweep, slos.get(wl, {})) if sweep else None
        servers = r.get("servers") or [{}]
        acc = quality.get(r["quality"], {}).get("score", {}).get("exec_accuracy", NAN)
        table.append({
            "experiment": name,
            "status": r["status"],
            "model": r["model"],
            "change": ", ".join(f"{k}={v}" for k, v in r["set"].items()) or "-",
            "num_gpus": r["num_gpus"],
            "exec_accuracy": acc,
            "kv_cache_tokens": sum(s.get("kv_cache_tokens", 0) for s in servers) or NAN,
            "weights_gib": servers[0].get("weights_gib", NAN),
            "ready_s": servers[0].get("ready_s", NAN),
            "peak_out_tok_s": max((x["output_tok_per_s"] for x in sweep), default=NAN),
            "slo_max_c": best["concurrency"] if best else 0,
            "slo_out_tok_s": best["output_tok_per_s"] if best else NAN,
            "slo_rps": best["request_throughput_rps"] if best else NAN,
            "usd_per_1m_tokens": usd_per_million_tokens(
                best["output_tok_per_s"] if best else NAN, r["num_gpus"], cost["usd_per_gpu_hour"]),
            # a SQL answer is ~25 tokens, so per query is the unit a product owner prices in
            "usd_per_1k_requests": usd_per_million_tokens(
                best["request_throughput_rps"] if best else NAN, r["num_gpus"],
                cost["usd_per_gpu_hour"]) / 1000,
            # acceptance depends on the text, not the load; report the lowest level's
            "spec_acceptance": sweep[0].get("spec_acceptance_rate", NAN) if sweep else NAN,
        })
    return table


def _fmt(v, spec: str) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return format(v, spec) if spec else str(v)


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(h for _, h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(_fmt(r[k], f) for k, _, f in cols) + " |\n" for r in rows)


VARIANT_COLS = [
    ("variant", "Variant", ""), ("n", "n", ""), ("exec_accuracy", "Exec acc", ".1%"),
    ("delta_pts", "Δ pts vs fp16", "+.1f"), ("mcnemar_p", "McNemar p", ".2g"),
    ("valid_sql_rate", "Valid SQL", ".1%"), ("size_gib", "Size GiB", ".2f"),
    ("compression_x", "× smaller", ".1f"), ("ppl_wikitext", "PPL wiki", ".2f"),
    ("ppl_wikitext_delta_pct", "Δ%", "+.1f"), ("ppl_spider", "PPL SQL", ".3f"),
    ("ppl_spider_delta_pct", "Δ%", "+.1f"), ("quantize_time_s", "Quant s", ".0f"),
    ("load_time_s", "Load s", ".0f"),
]
SERVING_COLS = [
    ("experiment", "Experiment", ""), ("status", "Status", ""), ("change", "Change", ""),
    ("num_gpus", "GPUs", ""), ("exec_accuracy", "Exec acc", ".1%"),
    ("weights_gib", "Weights GiB", ".2f"), ("kv_cache_tokens", "KV tokens", ",.0f"),
    ("ready_s", "Ready s", ".0f"), ("peak_out_tok_s", "Peak tok/s", ".0f"),
    ("slo_max_c", "Max c @SLO", ""), ("slo_out_tok_s", "tok/s @SLO", ".0f"),
    ("slo_rps", "req/s @SLO", ".2f"), ("usd_per_1m_tokens", "$/1M out tok", ".3f"),
    ("usd_per_1k_requests", "$/1k queries", ".4f"),
    ("spec_acceptance", "Spec accept", ".0%"),
]


def plot_pareto(table: list[dict], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in table if r["status"] == "ok"]
    front = pareto_front(pts, "usd_per_1m_tokens", "exec_accuracy")
    on = {r["experiment"] for r in front}
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    plotted = sorted((r for r in pts if r["usd_per_1m_tokens"] == r["usd_per_1m_tokens"]
                      and r["exec_accuracy"] == r["exec_accuracy"]), key=lambda r: r["usd_per_1m_tokens"])
    # Serving configs of one model share its accuracy and sit on one row: stagger their labels.
    offsets = [(6, 4), (6, -11), (6, 15)]
    row_count: dict[float, int] = {}
    for r in plotted:
        frontier = r["experiment"] in on
        y = r["exec_accuracy"] * 100
        k = row_count.get(round(y, 1), 0)
        row_count[round(y, 1)] = k + 1
        ax.plot(r["usd_per_1m_tokens"], y, marker="o", markersize=8 if frontier else 6,
                color=SERIES[0] if frontier else MUTED, markeredgecolor=SURFACE,
                markeredgewidth=1.5, linestyle="none", zorder=3)
        ax.annotate(r["experiment"], (r["usd_per_1m_tokens"], y), textcoords="offset points",
                    xytext=offsets[k % len(offsets)], fontsize=8, color=INK if frontier else MUTED)
    ax.plot([r["usd_per_1m_tokens"] for r in front], [r["exec_accuracy"] * 100 for r in front],
            color=SERIES[0], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
            zorder=2, label="Pareto frontier (not beaten on both cost and accuracy)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    _style(ax, "GPU cost per 1M output tokens at SLO capacity (USD, spider_v1)",
           "Spider dev execution accuracy (%)")
    ax.set_title("Accuracy vs serving cost (up and left is better)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    path = out / "pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiments", type=Path, required=True)
    p.add_argument("--quality", type=Path, nargs="+", required=True)
    p.add_argument("--plan", default="configs/experiments.yaml")
    p.add_argument("--workloads-config", default="configs/workloads.yaml")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    plan = yaml.safe_load(Path(a.plan).read_text())
    slos = yaml.safe_load(Path(a.workloads_config).read_text()).get("slo", {})
    quality = load_quality(a.quality)
    results, rows = load_experiments(a.experiments)
    order = {name: i for i, name in enumerate(plan["experiments"])}
    results = dict(sorted(results.items(), key=lambda kv: order.get(kv[0], len(order))))
    a.out.mkdir(parents=True, exist_ok=True)

    variants = variant_rows(quality)
    (a.out / "variants.md").write_text(md_table(variants, VARIANT_COLS))
    table = serving_rows(results, rows, quality, slos, plan["cost"])
    (a.out / "serving.md").write_text(md_table(table, SERVING_COLS))
    detail = detail_rows(rows)
    if detail:
        with (a.out / "detail.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0]))
            w.writeheader()
            w.writerows(detail)
    print((a.out / "variants.md").read_text())
    print((a.out / "serving.md").read_text())
    if table:
        print("wrote", plot_pareto(table, a.out))


if __name__ == "__main__":
    main()
