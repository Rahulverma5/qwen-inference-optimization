"""Export the merged fp16 model to GGUF for CPU serving with llama.cpp.

    python -m quantize.gguf --model models/fp16 --out models/gguf --llama-cpp build/llama.cpp

Clones llama.cpp at the tag pinned in configs/quant.yaml and builds only the CPU tools it
needs (llama-quantize, llama-server), converts HF -> f16 GGUF, then quantizes to each type
(Q8_0, Q4_K_M). The f16 intermediate is deleted; gguf_info.json records sizes and times.

The converter runs against the gguf-py shipped in the same checkout (PYTHONPATH), not the
pip `gguf` that vLLM installs: converter and gguf library must come from the same commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

GIB = 1024**3


def setup_commands(repo: str, ref: str, llama_dir: Path, jobs: int) -> list[list[str]]:
    build = llama_dir / "build"
    return [
        ["git", "clone", "--depth", "1", "--branch", ref, repo, str(llama_dir)],
        # CPU-only: these tools quantize and serve on CPU; a CUDA build takes ~20 min longer.
        ["cmake", "-S", str(llama_dir), "-B", str(build), "-DGGML_CUDA=OFF",
         "-DLLAMA_OPENSSL=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--target", "llama-quantize", "llama-server",
         "-j", str(jobs)],
    ]


def gguf_name(stem: str, qtype: str) -> str:
    return f"{stem}-{qtype}.gguf"


def export_commands(llama_dir: Path, model: Path, out: Path, stem: str,
                    types: list[str]) -> list[list[str]]:
    f16 = out / gguf_name(stem, "f16")
    cmds = [[sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(model),
             "--outtype", "f16", "--outfile", str(f16)]]
    quantize = llama_dir / "build" / "bin" / "llama-quantize"
    cmds += [[str(quantize), str(f16), str(out / gguf_name(stem, t)), t] for t in types]
    return cmds


def run(cmd: list[str], env: dict | None = None) -> float:
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="merged fp16 checkpoint dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llama-cpp", type=Path, required=True, help="checkout/build dir")
    p.add_argument("--config", default="configs/quant.yaml")
    p.add_argument("--stem", default="qwen3-1.7b-spider")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())["gguf"]
    info: dict = {"llama_cpp_ref": cfg["llama_cpp_ref"], "files": {}}
    if not (a.llama_cpp / "build" / "bin" / "llama-quantize").exists():
        t = sum(run(c) for c in setup_commands(cfg["llama_cpp_repo"], cfg["llama_cpp_ref"],
                                               a.llama_cpp, os.cpu_count() or 4))
        info["build_time_s"] = round(t, 1)

    a.out.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONPATH": str(a.llama_cpp / "gguf-py")}
    convert, *quants = export_commands(a.llama_cpp, a.model, a.out, a.stem, cfg["types"])
    info["convert_time_s"] = round(run(convert, env), 1)
    for qtype, cmd in zip(cfg["types"], quants, strict=True):
        dt = run(cmd)
        path = Path(cmd[2])
        info["files"][qtype] = {"file": path.name, "size_gib": round(path.stat().st_size / GIB, 3),
                                "quantize_time_s": round(dt, 1)}
    Path(convert[-1]).unlink()  # f16 GGUF duplicates the HF checkpoint
    (a.out / "gguf_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
