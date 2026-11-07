"""Calibration set for llm-compressor: a fixed sample of Spider *train* in the SFT format.

Calibration data should look like serving traffic: GPTQ minimizes layer output error on these
activations, and AWQ/SmoothQuant pick their scales from them. Using the same chat template,
schema-first prompt and gold SQL completion as SFT (finetune.train_sft.build_rows) means the
statistics come from the distribution the model actually serves. Dev is never used here, so
quantized accuracy on dev stays an honest held-out number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from evaluation.spider import db_path, schema_ddl
from finetune.prompt import build_messages, render


def sample_examples(examples: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample without replacement (all of them if n >= len)."""
    if n >= len(examples):
        return list(examples)
    return [examples[i] for i in sorted(random.Random(seed).sample(range(len(examples)), n))]


def calibration_texts(spider_dir: Path, splits: list[str], tokenizer, n: int, seed: int) -> list[str]:
    examples = [ex for s in splits for ex in json.loads((spider_dir / s).read_text())]
    return [
        render(tokenizer, build_messages(schema_ddl(db_path(spider_dir, ex["db_id"])), ex["question"]),
               add_generation_prompt=True) + ex["query"].strip() + tokenizer.eos_token
        for ex in sample_examples(examples, n, seed)
    ]


def tokenize(texts: list[str], tokenizer, max_len: int) -> dict[str, list[list[int]]]:
    """Unpadded token ids (llm-compressor calibrates with batch size 1), truncated to max_len."""
    ids = [tokenizer.encode(t, add_special_tokens=False)[:max_len] for t in texts]
    return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
