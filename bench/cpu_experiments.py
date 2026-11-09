"""CPU serving experiments: GGUF on llama-server (configs/cpu_experiments.yaml).

    python -m bench.cpu_experiments --llama-server build/llama.cpp/build/bin/llama-server \
        --gguf-dir models/gguf --workloads data/workloads --out reports/week3/cpu [--only q8_0]
    python -m bench.cpu_experiments ... --quality q4_k_m --spider-dir data/spider_data \
        --quality-out reports/week3/cpu/preds    # Spider accuracy of one config, served live

Same client, metrics and result layout as bench.experiments (so bench.matrix reads both), with
one difference: every (workload, level) gets a fresh server. llama.cpp keeps prompts in its
slots and, with --cache-ram, in host RAM; a restart is the only reset that covers both, and a
1-2 GB GGUF is mmap'd back in about a second.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from bench.server import launch_cmd, stop

PORT = 8080


def build_command(binary: str, model_file: Path, flags: dict, served: str, port: int) -> list[str]:
    cmd = [binary, "--model", str(model_file), "--alias", served, "--port", str(port)]
    for k, v in flags.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False:
            cmd.append(f"--no-{k}")
        elif v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd


def expand(plan: dict, only: list[str] | None = None) -> list[dict]:
    out = []
    for name, exp in plan["experiments"].items():
        if only and name not in only:
            continue
        levels = plan["levels"] | exp.get("levels", {})
        model = exp.get("model", "q4_k_m")
        out.append({
            "name": name, "model": model, "set": exp.get("set", {}),
            "flags": plan["baseline"] | exp.get("set", {}),
            "workloads": {wl: levels[wl] for wl in exp["workloads"]},
            "quality": exp.get("quality", model), "question": exp.get("question", ""),
        })
    if only and (missing := set(only) - {e["name"] for e in out}):
        raise ValueError(f"unknown experiments: {sorted(missing)}")
    return out


def client_command(name: str, workload: str, level: int, served: str, workloads_dir: Path,
                   out: Path, min_requests: int, waves: int) -> list[str]:
    cmd = [sys.executable, "-m", "bench.client", "--url", f"http://localhost:{PORT}",
           "--model", served, "--prompts", str(workloads_dir / f"{workload}.jsonl"),
           "--concurrency", str(level), "--min-requests", str(min_requests), "--waves", str(waves),
           "--warmup", "2", "--fresh-server", "--label", f"{name}/{workload}", "--out", str(out)]
    if workload.startswith("spider"):  # natural output length
        cmd.append("--no-ignore-eos")
    return cmd


def merge_summaries(level_dirs: list[Path], dest: Path) -> None:
    """One summary.csv per workload (the layout bench.matrix reads) from per-level client runs."""
    rows = []
    for d in level_dirs:
        if (d / "summary.csv").exists():
            with (d / "summary.csv").open() as f:
                rows += list(csv.DictReader(f))
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_experiment(exp: dict, plan: dict, binary: str, gguf_dir: Path, workloads_dir: Path,
                   out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = {k: exp[k] for k in ("name", "model", "set", "quality", "question")}
    # "servers" = the one server shape (what bench.matrix sums over replicas); every per-level
    # restart's startup facts go to "launches"
    result |= {"engine": "llama.cpp", "flags": exp["flags"], "num_units": 1, "servers": [],
               "launches": [], "workloads": {}}
    model_file = gguf_dir / plan["models"][exp["model"]]
    cmd = build_command(binary, model_file, exp["flags"], plan["served_model_name"], PORT)
    t0 = time.time()
    try:
        for wl, levels in exp["workloads"].items():
            level_dirs, failed = [], []
            for c in levels:
                proc, facts = launch_cmd(cmd, PORT, out / f"server_{wl}_c{c}.log", timeout_s=300)
                result["launches"].append(facts | {"workload": wl, "level": c})
                result["servers"] = result["servers"] or [facts]
                if not facts["ready"]:
                    result["status"] = "server_failed"
                    result["error_tail"] = (out / f"server_{wl}_c{c}.log").read_text()[-4000:]
                    return result
                try:
                    level_dirs.append(out / wl / f"c{c}")
                    rc = subprocess.run(client_command(
                        exp["name"], wl, c, plan["served_model_name"], workloads_dir, level_dirs[-1],
                        plan["min_requests"], plan["waves"])).returncode
                finally:
                    stop(proc)
                if rc != 0:
                    failed.append(c)
            merge_summaries(level_dirs, out / wl / "summary.csv")
            result["workloads"][wl] = "ok" if not failed else f"failed at c={failed}"
        bad = [w for w, s in result["workloads"].items() if s != "ok"]
        result["status"] = "ok" if not bad else "client_failed"
        return result
    finally:
        result["wall_s"] = round(time.time() - t0, 1)
        (out / "result.json").write_text(json.dumps(result, indent=2))


def run_quality(exp: dict, plan: dict, binary: str, gguf_dir: Path, tokenizer: str,
                spider_dir: Path, out: Path, limit: int | None) -> int:
    """Spider exec accuracy of one config served live (evaluation.remote), slots = concurrency."""
    cmd = build_command(binary, gguf_dir / plan["models"][exp["model"]], exp["flags"],
                        plan["served_model_name"], PORT)
    proc, facts = launch_cmd(cmd, PORT, out / f"server_quality_{exp['name']}.log", timeout_s=300)
    if not facts["ready"]:
        return 1
    try:
        ev = [sys.executable, "-m", "evaluation.remote", "--url", f"http://localhost:{PORT}",
              "--model", plan["served_model_name"], "--tokenizer", tokenizer,
              "--spider-dir", str(spider_dir), "--concurrency", str(exp["flags"].get("parallel", 1)),
              "--out", str(out / f"{exp['quality']}.jsonl")]
        return subprocess.run(ev + (["--limit", str(limit)] if limit else [])).returncode
    finally:
        stop(proc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--plan", default="configs/cpu_experiments.yaml")
    p.add_argument("--llama-server", required=True)
    p.add_argument("--gguf-dir", type=Path, required=True)
    p.add_argument("--workloads", type=Path, help="dir of bench.workloads JSONL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quality", default=None, metavar="EXPERIMENT",
                   help="instead of benchmarking, measure Spider accuracy of this config")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--spider-dir", type=Path)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    plan = yaml.safe_load(Path(a.plan).read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    if a.quality:
        (exp,) = expand(plan, [a.quality])
        sys.exit(run_quality(exp, plan, a.llama_server, a.gguf_dir, a.tokenizer, a.spider_dir,
                             a.out, a.limit))
    statuses = {}
    for exp in expand(plan, a.only):
        done = a.out / exp["name"] / "result.json"
        if done.exists() and not a.force and json.loads(done.read_text()).get("status") == "ok":
            statuses[exp["name"]] = "ok (cached)"
            continue
        print(f"=== {exp['name']}: {exp['question']}", flush=True)
        r = run_experiment(exp, plan, a.llama_server, a.gguf_dir, a.workloads, a.out / exp["name"])
        statuses[exp["name"]] = r["status"]
        print(f"=== {exp['name']}: {r['status']} in {r['wall_s']:.0f} s", flush=True)
    (a.out / "status.json").write_text(json.dumps(statuses, indent=2))
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
