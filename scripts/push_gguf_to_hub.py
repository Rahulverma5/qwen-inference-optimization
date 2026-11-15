"""Publish the deployment GGUF (Week 3 Q4_K_M) to the Hugging Face Hub with a model card.

    hf auth login                     # a token with write access
    python scripts/push_gguf_to_hub.py --repo <hf-user>/qwen3-1.7b-spider-sql-GGUF [--private]

Prints the commit hash to pin as `hf_revision` in infra/aws/terraform.tfvars, so a redeploy
always pulls the exact bytes that were evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi

GGUF = Path("models/week3/models/gguf/qwen3-1.7b-spider-Q4_K_M.gguf")

CARD = """---
base_model: Qwen/Qwen3-1.7B
license: apache-2.0
datasets: [xlangai/spider]
language: [en]
pipeline_tag: text-generation
tags: [text-to-sql, gguf, llama.cpp, qlora, spider]
---

# Qwen3-1.7B fine-tuned for text-to-SQL (Spider), GGUF Q4_K_M

QLoRA fine-tune of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) on the Spider train
split, merged and quantized to GGUF **Q4_K_M** (1.03 GiB) with llama.cpp b11600. Built on free
Kaggle T4s; served on CPU. Code, training and evaluation:
[github.com/Rahulverma5/qwen-inference-optimization](https://github.com/Rahulverma5/qwen-inference-optimization).

| Spider dev (1,034 questions), execution accuracy | |
|---|---|
| Qwen3-1.7B base | 55.9% |
| Fine-tuned, fp16 | 64.7% |
| **This file, served by llama-server on CPU** | **64.3%** (−0.4 vs fp16, McNemar p = 0.76) |

`sha256: {sha256}`

## Prompt format

ChatML with thinking **off**. System: `{system}`
User: `Schema:\\n<CREATE TABLE ...>\\n\\nQuestion: <question>`.
The schema comes first so the server's prompt cache can reuse it across questions on one database.

```bash
llama-server -m qwen3-1.7b-spider-Q4_K_M.gguf --parallel 1 --ctx-size 2048 --cache-ram 1024 --flash-attn on
```

Then send the rendered prompt (ending `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`)
to `/v1/completions` with `temperature 0`.

## Limitations

Trained and evaluated on Spider only (SQLite, English); expect lower accuracy on other SQL dialects
and messy real schemas. About 1 in 3 dev answers is wrong: review generated SQL before running it
on data you care about, and run it with read-only credentials. Spider is CC BY-SA 4.0.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    from finetune.prompt import SYSTEM

    h = hashlib.sha256()
    with GGUF.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.format(sha256=h.hexdigest(), system=SYSTEM).encode(),
                    path_in_repo="README.md", repo_id=a.repo, commit_message="Model card")
    info = api.upload_file(path_or_fileobj=str(GGUF), path_in_repo=GGUF.name, repo_id=a.repo,
                           commit_message="Q4_K_M GGUF (Week 3)")
    print(f"uploaded https://huggingface.co/{a.repo}\nsha256 {h.hexdigest()}\nhf_revision = \"{info.oid}\"")


if __name__ == "__main__":
    main()
