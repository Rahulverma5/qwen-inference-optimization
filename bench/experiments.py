"""Run the Week 3 serving experiments in configs/experiments.yaml, one server config at a time.

    python -m bench.experiments --models-dir /kaggle/working/models --workloads data/workloads \
        --out reports/week3/experiments [--only fp16 tp2] [--force]

Per experiment: start its server(s) on the right GPUs, run bench.client over each workload,
stop the servers, write <out>/<name>/result.json. A server that fails to start (an
unsupported flag on sm75 is a result, not a crash) is recorded with the tail of its log, and
the run moves on. Finished experiments are skipped on re-run, so a Kaggle session that dies
can resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from bench.server import launch, stop

BASE_PORT = 8000


def expand(plan: dict, only: list[str] | None = None) -> list[dict]:
    """Experiment specs with defaults filled in, in file order."""
    out = []
    for name, exp in plan["experiments"].items():
        if only and name not in only:
            continue
        levels = plan["levels"] | exp.get("levels", {})
        model = exp.get("model", "fp16")
        out.append({
            "name": name,
            "model": model,
            "set": exp.get("set", {}),
            "replicas": exp.get("replicas", 1),
            "gpus_per_replica": exp.get("gpus_per_replica", 1),
            "workloads": {wl: levels[wl] for wl in exp["workloads"]},
            "quality": exp.get("quality", model),
            "question": exp.get("question", ""),
        })
    if only and (missing := set(only) - {e["name"] for e in out}):
        raise ValueError(f"unknown experiments: {sorted(missing)}")
    return out


def server_specs(exp: dict, base_profile: dict) -> list[dict]:
    """One entry per replica: its GPUs, port, and flags. Replicas get disjoint GPUs."""
    g = exp["gpus_per_replica"]
    profile = base_profile | exp["set"] | ({"tensor-parallel-size": g} if g > 1 else {})
    return [{"gpus": ",".join(str(i * g + j) for j in range(g)), "port": BASE_PORT + i,
             "profile": profile} for i in range(exp["replicas"])]


def client_command(exp: dict, workload: str, levels: list[int], urls: list[str], served: str,
                   workloads_dir: Path, out: Path, min_requests: int, waves: int) -> list[str]:
    cmd = [sys.executable, "-m", "bench.client", "--url", *urls, "--model", served,
           "--prompts", str(workloads_dir / f"{workload}.jsonl"),
           "--concurrency", *map(str, levels), "--min-requests", str(min_requests),
           "--waves", str(waves), "--label", f"{exp['name']}/{workload}", "--out", str(out)]
    if workload.startswith("spider"):  # natural output length: what production traffic does
        cmd.append("--no-ignore-eos")
    return cmd


def model_path(models_dir: Path, variant: str) -> Path:
    path = models_dir / variant
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"no model for variant {variant!r} at {path}")
    return path


def run_experiment(exp: dict, plan: dict, base_profile: dict, models_dir: Path,
                   workloads_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = {k: exp[k] for k in ("name", "model", "set", "replicas", "gpus_per_replica",
                                  "quality", "question")}
    result["num_gpus"] = exp["replicas"] * exp["gpus_per_replica"]
    result["servers"], result["workloads"] = [], {}
    procs = []
    t0 = time.time()
    try:
        model = str(model_path(models_dir, exp["model"]))
        for i, spec in enumerate(server_specs(exp, base_profile)):
            log = out / f"server{i}.log"
            proc, facts = launch(model, spec["profile"], plan["served_model_name"], spec["port"],
                                 log, env={"CUDA_VISIBLE_DEVICES": spec["gpus"]})
            result["servers"].append(facts | {"gpus": spec["gpus"], "port": spec["port"]})
            if not facts["ready"]:
                result["status"] = "server_failed"
                result["error_tail"] = log.read_text()[-4000:]
                return result
            procs.append(proc)
        urls = [f"http://localhost:{s['port']}" for s in result["servers"]]
        for wl, levels in exp["workloads"].items():
            cmd = client_command(exp, wl, levels, urls, plan["served_model_name"], workloads_dir,
                                 out / wl, plan["min_requests"], plan["waves"])
            print("$", " ".join(cmd), flush=True)
            rc = subprocess.run(cmd).returncode
            result["workloads"][wl] = "ok" if rc == 0 else f"failed (exit {rc})"
        failed = [w for w, s in result["workloads"].items() if s != "ok"]
        result["status"] = "ok" if not failed else "client_failed"
        return result
    except FileNotFoundError as e:
        result["status"] = "missing_model"
        result["error_tail"] = str(e)
        return result
    finally:
        for proc in procs:
            stop(proc)
        result["wall_s"] = round(time.time() - t0, 1)
        (out / "result.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--plan", default="configs/experiments.yaml")
    p.add_argument("--serving", default="configs/serving.yaml")
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--workloads", type=Path, required=True, help="dir of bench.workloads JSONL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None, help="run just these experiments")
    p.add_argument("--force", action="store_true", help="re-run experiments that already finished")
    a = p.parse_args()

    plan = yaml.safe_load(Path(a.plan).read_text())
    base_profile = yaml.safe_load(Path(a.serving).read_text())["profiles"]["baseline"]
    a.out.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for exp in expand(plan, a.only):
        done = a.out / exp["name"] / "result.json"
        if done.exists() and not a.force and json.loads(done.read_text()).get("status") == "ok":
            statuses[exp["name"]] = "ok (cached)"
            continue
        print(f"=== {exp['name']}: {exp['question']}", flush=True)
        r = run_experiment(exp, plan, base_profile, a.models_dir, a.workloads, a.out / exp["name"])
        statuses[exp["name"]] = r["status"]
        print(f"=== {exp['name']}: {r['status']} in {r['wall_s']:.0f} s", flush=True)
    (a.out / "status.json").write_text(json.dumps(statuses, indent=2))
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
