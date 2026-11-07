"""Launch a vLLM OpenAI-compatible server from a profile in configs/serving.yaml.

    python -m bench.server --model /kaggle/working/sft-merged --profile baseline \
        --log reports/week2/server.log [--set max-num-seqs=64]

Blocks until /health answers (or the process dies), prints the KV capacity vLLM reports, and
leaves the server running in the background; its PID goes to <log>.pid, and startup facts
(ready time, weight memory, KV capacity) to <log>.ready.json. bench.experiments uses
launch()/stop() directly to run one server per experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

# vLLM V1 startup lines, e.g.
#   GPU KV cache size: 81,456 tokens
#   Maximum concurrency for 4,096 tokens per request: 19.89x
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_MAX_CONC = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
#   Model loading took 3.2 GiB memory and 5.123 seconds
_LOAD = re.compile(r"Model loading took\s*([\d.]+)\s*GiB(?: memory)? and\s*([\d.]+)\s*seconds")


def parse_kv_capacity(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if m := _KV_TOKENS.search(log):
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := _MAX_CONC.search(log):
        out["max_concurrency_seq_len"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    return out


def parse_model_load(log: str) -> dict[str, float]:
    """Weight memory and weight-loading time per GPU worker (the first worker's line)."""
    if m := _LOAD.search(log):
        return {"weights_gib": float(m.group(1)), "weights_load_s": float(m.group(2))}
    return {}


def build_command(model: str, profile: dict, served_name: str, port: int) -> list[str]:
    cmd = ["vllm", "serve", model, "--served-model-name", served_name, "--port", str(port)]
    for k, v in profile.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False:
            cmd.append(f"--no-{k}")
        elif isinstance(v, dict):  # e.g. --speculative-config takes JSON
            cmd += [f"--{k}", json.dumps(v)]
        elif v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd


def parse_overrides(pairs: list[str]) -> dict:
    return {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in pairs)}


def wait_ready(url: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(3)
    return False


def launch(model: str, profile: dict, served_name: str, port: int, log_path: Path,
           timeout_s: float = 900, env: dict | None = None) -> tuple[subprocess.Popen, dict]:
    """Start a server in its own session and wait for /health. Returns (process, startup facts);
    facts["ready"] is False if it died or timed out (the process is then already stopped)."""
    cmd = build_command(model, profile, served_name, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".cmd").write_text(" ".join(cmd) + "\n")
    print("$", " ".join(cmd), flush=True)
    env = os.environ | {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # same JIT problem as the attention backend
        "VLLM_SERVER_DEV_MODE": "1",  # exposes /reset_prefix_cache, used by bench.client per level
    } | (env or {})
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)  # survives this script exiting
    log_path.with_suffix(".pid").write_text(f"{proc.pid}\n")
    t0 = time.time()
    ok = wait_ready(f"http://localhost:{port}", proc, timeout_s)
    text = log_path.read_text()
    facts = {"ready": ok, "ready_s": round(time.time() - t0, 1)} | parse_kv_capacity(text) \
        | parse_model_load(text)
    if not ok:
        facts["exit_code"] = proc.poll()
        stop(proc)
    log_path.with_suffix(".ready.json").write_text(json.dumps(facts, indent=2))
    return proc, facts


def stop(proc: subprocess.Popen, timeout_s: float = 60) -> None:
    """SIGTERM the server's whole process group (API server + engine core + workers), then
    SIGKILL if it hasn't exited, so the GPU memory is free for the next experiment."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="configs/serving.yaml")
    p.add_argument("--profile", default="baseline")
    p.add_argument("--set", nargs="*", default=[], metavar="FLAG=VALUE",
                   help="override profile flags, e.g. max-num-seqs=64 enable-prefix-caching=false")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=900, help="seconds to wait for /health")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    profile = cfg["profiles"][a.profile] | parse_overrides(a.set)
    proc, facts = launch(a.model, profile, cfg["served_model_name"], a.port, a.log, a.timeout)
    if not facts["ready"]:
        print(a.log.read_text()[-3000:])
        sys.exit(f"server not ready after {facts['ready_s']:.0f} s (exit code {facts['exit_code']})")
    print(f"ready in {facts['ready_s']:.0f} s at http://localhost:{a.port}; {facts}")


if __name__ == "__main__":
    main()
