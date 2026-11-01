"""Turn sweep summaries into capacity numbers and plots.

    python -m bench.report reports/week2/kaggle/results/runs/*/summary.csv --out reports/week2

For each closed-loop sweep:
  knee      - the concurrency maximizing Kleinrock's "power" = throughput / latency: past it,
              extra load buys less throughput than it costs in latency.
  SLO max   - the highest concurrency whose p95s meet the workload's SLO (configs/workloads.yaml).
  Little    - L = lambda * W: concurrency should equal req/s x mean E2E. A ratio well below 1
              means the client didn't keep N requests in flight (client-bound or tail drain).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import yaml

# Reference palette (dataviz skill), categorical slots in fixed order; ink tokens for text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
# Color follows the workload, not its position in the file list, so plots stay comparable.
WORKLOAD_ORDER = ["short_v1", "long_v1", "shared_prefix_v1", "spider_v1"]


def load_summaries(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open() as f:
            for r in csv.DictReader(f):
                rows.append({k: _num(v) for k, v in r.items()})
    return rows


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def closed_loop(rows: list[dict], label: str) -> list[dict]:
    """Closed-loop rows of one label, sorted by concurrency, with `concurrency` parsed from `c8`."""
    out = [r | {"concurrency": int(r["mode"][1:])} for r in rows
           if r["label"] == label and str(r["mode"]).startswith("c")]
    return sorted(out, key=lambda r: r["concurrency"])


def meets_slo(row: dict, slo: dict) -> bool:
    return row["error_rate"] == 0 and all(row[k] <= v for k, v in slo.items())


def max_under_slo(rows: list[dict], slo: dict) -> dict | None:
    """Highest concurrency that meets the SLO with every lower level also meeting it, so a
    lucky pass above a failing level isn't reported as capacity."""
    best = None
    for r in rows:
        if not meets_slo(r, slo):
            break
        best = r
    return best


def knee(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["output_tok_per_s"] / r["e2e_s_mean"])


def littles_ratio(row: dict) -> float:
    return row["request_throughput_rps"] * row["e2e_s_mean"] / row["concurrency"]


def capacity_table(rows: list[dict], slos: dict) -> list[dict]:
    table = []
    for label in dict.fromkeys(r["label"] for r in rows):
        sweep = closed_loop(rows, label)
        if not sweep:
            continue
        slo = slos.get(label.split("/")[-1], {})
        k, s = knee(sweep), max_under_slo(sweep, slo) if slo else None
        table.append({
            "workload": label,
            "peak_out_tok_s": max(r["output_tok_per_s"] for r in sweep),
            "knee_c": k["concurrency"],
            "knee_out_tok_s": k["output_tok_per_s"],
            "slo": ", ".join(f"{m} <= {v * 1e3:.0f} ms" for m, v in slo.items()) or "-",
            "slo_max_c": s["concurrency"] if s else 0,
            "slo_out_tok_s": s["output_tok_per_s"] if s else 0.0,
            "slo_rps": s["request_throughput_rps"] if s else 0.0,
            "littles_min": min(littles_ratio(r) for r in sweep),
            "kv_usage_peak": max((r.get("kv_usage_max", math.nan) for r in sweep), default=math.nan),
        })
    return table


def to_markdown(table: list[dict]) -> str:
    head = ("| Workload | Peak out tok/s | Knee c | tok/s at knee | SLO | Max c under SLO | "
            "tok/s / req/s at SLO | Little min | Peak KV use |\n|---|---|---|---|---|---|---|---|---|\n")
    return head + "".join(
        f"| {t['workload']} | {t['peak_out_tok_s']:.0f} | {t['knee_c']} | {t['knee_out_tok_s']:.0f} | "
        f"{t['slo']} | {t['slo_max_c']} | {t['slo_out_tok_s']:.0f} / {t['slo_rps']:.2f} | "
        f"{t['littles_min']:.2f} | {t['kv_usage_peak']:.0%} |\n" for t in table)


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def plot(rows: list[dict], slos: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def slot(label: str) -> int:
        wl = label.split("/")[-1]
        return WORKLOAD_ORDER.index(wl) if wl in WORKLOAD_ORDER else len(WORKLOAD_ORDER)

    labels = sorted((lb for lb in dict.fromkeys(r["label"] for r in rows) if closed_loop(rows, lb)),
                    key=slot)
    if len(labels) > len(SERIES):
        raise ValueError(f"{len(labels)} series; split into several reports (max {len(SERIES)})")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    paths = []

    # 1. Throughput vs per-token latency: the trade-off curve; the knee is where it bends up.
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    for color, lb in zip(SERIES, labels, strict=False):
        sw = closed_loop(rows, lb)
        x = [r["output_tok_per_s"] for r in sw]
        y = [r["tpot_s_p95"] * 1e3 for r in sw]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lb)
        # Label only the high-load end; low-load points of all workloads overlap.
        ax.annotate(f"c={sw[-1]['concurrency']}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=MUTED)
        k = knee(sw)
        ax.plot(k["output_tok_per_s"], k["tpot_s_p95"] * 1e3, marker="D", markersize=9,
                color=color, markeredgecolor=INK, markeredgewidth=1, linestyle="none")
    tpot_slos = {s.get("tpot_s_p95") for s in slos.values()} - {None}
    for v in tpot_slos:
        ax.axhline(v * 1e3, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"TPOT SLO {v * 1e3:.0f} ms", (ax.get_xlim()[0], v * 1e3), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=MUTED)
    _style(ax, "Output throughput (tokens/s)", "TPOT p95 (ms)")
    ax.set_title("Throughput vs per-token latency (◆ = knee)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    paths.append(out / "throughput_vs_tpot.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 2. Latency vs concurrency, one panel per metric (never a second y-axis).
    metrics = [("ttft_s_p95", "TTFT p95 (ms)"), ("tpot_s_p95", "TPOT p95 (ms)"),
               ("e2e_s_p95", "E2E p95 (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=SURFACE)
    for ax, (m, title) in zip(axes, metrics, strict=True):
        for color, lb in zip(SERIES, labels, strict=False):
            sw = closed_loop(rows, lb)
            ax.plot([r["concurrency"] for r in sw], [r[m] * 1e3 for r in sw], color=color,
                    linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, label=lb)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(NullFormatter())
        _style(ax, "Concurrent requests", title)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Latency vs concurrency (closed loop, log–log)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    paths.append(out / "latency_vs_concurrency.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--config", default="configs/workloads.yaml")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rows = load_summaries(a.summaries)
    slos = yaml.safe_load(Path(a.config).read_text()).get("slo", {})
    a.out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(capacity_table(rows, slos))
    (a.out / "capacity.md").write_text(md)
    print(md)
    for path in plot(rows, slos, a.out):
        print("wrote", path)


if __name__ == "__main__":
    main()
