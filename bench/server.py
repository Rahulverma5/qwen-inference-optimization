"""Launch a vLLM OpenAI-compatible server from a profile in configs/serving.yaml.

    python -m bench.server --model /kaggle/working/sft-merged --profile baseline \
        --log reports/week2/server.log [--set max-num-seqs=64]

Blocks until /health answers (or the process dies), prints the KV capacity vLLM reports, and
leaves the server running in the background; its PID goes to <log>.pid.
"""

from __future__ import annotations

import argparse
import os
import re
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


def parse_kv_capacity(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if m := _KV_TOKENS.search(log):
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := _MAX_CONC.search(log):
        out["max_concurrency_seq_len"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    return out


def build_command(model: str, profile: dict, served_name: str, port: int) -> list[str]:
    cmd = ["vllm", "serve", model, "--served-model-name", served_name, "--port", str(port)]
    for k, v in profile.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False:
            cmd.append(f"--no-{k}")
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
    cmd = build_command(a.model, profile, cfg["served_model_name"], a.port)
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.log.with_suffix(".cmd").write_text(" ".join(cmd) + "\n")
    print("$", " ".join(cmd), flush=True)

    env = os.environ | {"VLLM_USE_FLASHINFER_SAMPLER": "0"}  # same JIT problem as the backend
    with a.log.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)  # survives this script exiting
    a.log.with_suffix(".pid").write_text(f"{proc.pid}\n")
    url = f"http://localhost:{a.port}"
    t0 = time.time()
    ok = wait_ready(url, proc, a.timeout)
    text = a.log.read_text()
    if not ok:
        print(text[-3000:])
        sys.exit(f"server not ready after {time.time() - t0:.0f} s (exit code {proc.poll()})")
    print(f"ready in {time.time() - t0:.0f} s at {url}; KV capacity: {parse_kv_capacity(text)}")


if __name__ == "__main__":
    main()
