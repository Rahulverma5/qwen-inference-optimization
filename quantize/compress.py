"""Quantize the merged fp16 model with llm-compressor (one variant from configs/quant.yaml).

    python -m quantize.compress --model models/fp16 --variant awq_w4a16 \
        --spider-dir data/spider_data --out models/awq_w4a16

Output is a compressed-tensors checkpoint that vLLM loads with no --quantization flag, plus
quant_info.json (recipe, calibration size, wall time, size on disk).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from quantize.calibration import calibration_texts, tokenize

GIB = 1024**3


def dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / GIB


def build_recipe(modifiers: list[dict]) -> list:
    """[{ModifierName: kwargs}, ...] -> llm-compressor modifier objects, in order."""
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    known = {c.__name__: c for c in (AWQModifier, GPTQModifier, SmoothQuantModifier)}
    recipe = []
    for entry in modifiers:
        ((name, kwargs),) = entry.items()
        if name not in known:
            raise ValueError(f"unknown modifier {name}; known: {sorted(known)}")
        recipe.append(known[name](**kwargs))
    return recipe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="merged fp16 checkpoint dir")
    p.add_argument("--variant", required=True)
    p.add_argument("--config", default="configs/quant.yaml")
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=None, help="override, e.g. 16 for a smoke test")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    cal, spec = cfg["calibration"], cfg["variants"][a.variant]
    n = a.num_samples or cal["num_samples"]

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    texts = calibration_texts(a.spider_dir, cal["splits"], tok, n, cal["seed"])
    ds = Dataset.from_dict(tokenize(texts, tok, cal["max_seq_length"]))
    n_tokens = sum(map(len, ds["input_ids"]))
    print(f"calibration: {len(ds)} samples, {n_tokens:,} tokens")

    # Loaded on CPU; llm-compressor's sequential pipeline moves one decoder layer at a time to
    # the GPU. fp16, because the T4 has no bf16 (the checkpoint dtype).
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    t0 = time.perf_counter()
    oneshot(model=model, dataset=ds, recipe=build_recipe(spec["modifiers"]),
            max_seq_length=cal["max_seq_length"], num_calibration_samples=len(ds))
    elapsed = time.perf_counter() - t0

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, save_compressed=True)
    tok.save_pretrained(a.out)
    info = {
        "variant": a.variant,
        "modifiers": spec["modifiers"],
        "calibration_samples": len(ds),
        "calibration_tokens": n_tokens,
        "quantize_time_s": round(elapsed, 1),
        "size_gib": round(dir_size_gib(a.out), 3),
        "source_size_gib": round(dir_size_gib(Path(a.model)), 3),
    }
    (a.out / "quant_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
