"""Perplexity of a (quantized) model with vLLM prompt logprobs, on two corpora:

  wikitext  general text (WikiText-2 test, non-overlapping 1,024-token windows): does
            quantization damage the language model in general?
  spider    gold SQL given the serving prompt (Spider dev): the task distribution, scored on the
            completion tokens only. More sensitive to what matters here than WikiText.

    python -m evaluation.perplexity --model models/awq_w4a16 --corpus wikitext spider \
        --spider-dir data/spider_data --out reports/week3/ppl/awq_w4a16.json

Every variant (fp16, compressed-tensors, GGUF) goes through the same engine and the same
windows, so differences are the model's, not the harness's. Non-overlapping windows give
higher absolute perplexity than a sliding window (early tokens of each window have little
context); only deltas between variants are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def windows(ids: list[int], size: int, max_windows: int | None = None) -> list[list[int]]:
    """Non-overlapping full windows; the ragged tail is dropped so every window is equal work."""
    out = [ids[i:i + size] for i in range(0, len(ids) - size + 1, size)]
    return out[:max_windows]


def nll_from_prompt_logprobs(token_ids: list[int], prompt_logprobs: list, start: int) -> tuple[float, int]:
    """Sum of -log p(token_i) for i >= start, from vLLM's per-position {token_id: Logprob} dicts.
    Position 0 has no prediction (None), so start must be >= 1."""
    if start < 1:
        raise ValueError("position 0 is never predicted; start must be >= 1")
    total = 0.0
    for i in range(start, len(token_ids)):
        entry = prompt_logprobs[i][token_ids[i]]
        total -= entry.logprob if hasattr(entry, "logprob") else entry
    return total, len(token_ids) - start


def perplexity(nll_sum: float, n_tokens: int) -> float:
    return math.exp(nll_sum / n_tokens)


def wikitext_items(tokenizer, size: int, max_windows: int | None) -> list[tuple[list[int], int]]:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return [(w, 1) for w in windows(tokenizer.encode(text, add_special_tokens=False), size, max_windows)]


def spider_items(tokenizer, spider_dir: Path, limit: int | None) -> list[tuple[list[int], int]]:
    from evaluation.spider import db_path, schema_ddl
    from finetune.prompt import build_messages, render

    items = []
    for ex in json.loads((spider_dir / "dev.json").read_text())[:limit]:
        prompt = render(tokenizer, build_messages(schema_ddl(db_path(spider_dir, ex["db_id"])),
                                                  ex["question"]), add_generation_prompt=True)
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        c_ids = tokenizer.encode(ex["query"].strip() + tokenizer.eos_token, add_special_tokens=False)
        items.append((p_ids + c_ids, len(p_ids)))
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None, help="needed for a .gguf model")
    p.add_argument("--hf-config-path", default=None, help="config dir for a .gguf model")
    p.add_argument("--corpus", nargs="+", default=["wikitext", "spider"], choices=["wikitext", "spider"])
    p.add_argument("--spider-dir", type=Path, default=None)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="spider: first N dev examples")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    # Prompt logprobs materialize full-vocab (152k) logits for every scheduled prompt token,
    # outside vLLM's memory profile: cap tokens per step and leave headroom.
    llm = LLM(model=a.model, tokenizer=a.tokenizer, hf_config_path=a.hf_config_path,
              dtype="float16", max_model_len=max(a.window, 2048), gpu_memory_utilization=0.7,
              max_num_batched_tokens=2048, max_num_seqs=16, enable_prefix_caching=False,
              attention_backend="TRITON_ATTN", seed=0)
    tok = llm.get_tokenizer()
    result: dict = {"model": a.model, "window": a.window}
    for corpus in a.corpus:
        items = (wikitext_items(tok, a.window, a.max_windows) if corpus == "wikitext"
                 else spider_items(tok, a.spider_dir, a.limit))
        outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in items],
                            SamplingParams(max_tokens=1, prompt_logprobs=0))
        nll = n = 0
        for (ids, start), o in zip(items, outs, strict=True):
            s, k = nll_from_prompt_logprobs(ids, o.prompt_logprobs, start)
            nll, n = nll + s, n + k
        result[corpus] = {"ppl": perplexity(nll, n), "tokens": n, "sequences": len(items)}
        print(corpus, result[corpus], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
