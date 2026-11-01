"""KV-cache capacity model.

bytes/token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes

Used to answer "how many concurrent N-token sequences fit on this GPU?" before running
anything, and to sanity-check the KV block count vLLM reports at startup.

    python -m bench.kv_cache --model Qwen/Qwen3-1.7B --gpu-mem-gib 15 --seq-len 4096 16384
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class AttnShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, cfg: dict) -> AttnShape:
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        return cls(
            num_layers=cfg["num_hidden_layers"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=head_dim,
        )


def kv_bytes_per_token(shape: AttnShape, kv_dtype: str = "float16") -> int:
    return 2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * DTYPE_BYTES[kv_dtype]


def kv_budget_bytes(
    gpu_mem_gib: float,
    weight_gib: float,
    gpu_memory_utilization: float = 0.9,
    activation_overhead_gib: float = 1.0,
) -> float:
    """Memory left for KV cache, mirroring vLLM's accounting:
    util * total - weights - peak activations (profiled by vLLM; here a rough constant)."""
    return max(0.0, gpu_memory_utilization * gpu_mem_gib - weight_gib - activation_overhead_gib) * GIB


def max_concurrent_seqs(budget_bytes: float, seq_len: int, bytes_per_token: int) -> int:
    return int(budget_bytes // (seq_len * bytes_per_token))


def load_hf_config(model: str) -> dict:
    path = Path(model) / "config.json"
    if path.exists():
        return json.loads(path.read_text())
    from huggingface_hub import hf_hub_download  # only needed for hub ids

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="HF hub id or local dir with config.json")
    p.add_argument("--gpu-mem-gib", type=float, default=15.0)
    p.add_argument("--weight-gib", type=float, required=True,
                   help="weights size on GPU, e.g. 3.4 for 1.7B fp16, ~1.2 for W4A16")
    p.add_argument("--kv-dtype", default="float16", choices=DTYPE_BYTES)
    p.add_argument("--util", type=float, default=0.9)
    p.add_argument("--activation-gib", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, nargs="+", default=[4096, 16384])
    a = p.parse_args()

    shape = AttnShape.from_hf_config(load_hf_config(a.model))
    bpt = kv_bytes_per_token(shape, a.kv_dtype)
    budget = kv_budget_bytes(a.gpu_mem_gib, a.weight_gib, a.util, a.activation_gib)
    print(f"{a.model}: {shape}")
    print(f"KV bytes/token ({a.kv_dtype}): {bpt:,} ({bpt / 1024:.0f} KiB)")
    print(f"KV budget: {budget / GIB:.2f} GiB -> {int(budget // bpt):,} tokens total")
    for L in a.seq_len:
        print(f"  seq_len={L:>6}: {L * bpt / GIB:.3f} GiB/seq, "
              f"max concurrent = {max_concurrent_seqs(budget, L, bpt)}")


if __name__ == "__main__":
    main()
