"""Merge a LoRA adapter into the fp16 base so it can be quantized and served as one model.

    python -m finetune.merge --base Qwen/Qwen3-1.7B --adapter models/sft/adapter \
        --out models/sft-merged [--push <hf-user>/qwen3-1.7b-spider-sft]

Merge into the *fp16* base, not the 4-bit one used for QLoRA training: merging into
dequantized NF4 weights bakes quantization noise into every downstream variant.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--push", default=None, help="HF Hub repo id to push the merged model to")
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(a.base)
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    if a.push:
        merged.push_to_hub(a.push, private=True)
        tok.push_to_hub(a.push, private=True)


if __name__ == "__main__":
    main()
