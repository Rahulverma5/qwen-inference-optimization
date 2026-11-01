"""Build frozen prompt sets from configs/workloads.yaml and write them to JSONL.

Generation is deterministic (fixed seed, pinned tokenizer), so every engine/config is measured
on byte-identical inputs. Each file's sha256 is written to MANIFEST.json and kept with the
results; compare manifests to confirm two weeks' runs used the same prompts.

    python -m bench.workloads --tokenizer Qwen/Qwen3-1.7B --out data/workloads \
        --spider-dir data/spider_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path

import yaml

# Neutral filler vocabulary; content doesn't matter for throughput, only token counts do.
_WORDS = (
    "system latency memory cache token batch request model layer kernel queue schedule "
    "throughput prefill decode attention matrix vector tensor compute bandwidth server "
    "client stream window budget profile metric sample value index table query result"
).split()


def _filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def take_tokens(text: str, target: int, encode: Callable[[str], list[int]]) -> list[int]:
    ids = encode(text)
    if len(ids) < target:
        raise ValueError(f"text has {len(ids)} tokens, need {target}; generate more filler")
    return ids[:target]


def make_prompts(spec: dict, seed: int, encode, decode) -> list[dict]:
    rng = random.Random(seed)
    n, p_tok, o_tok = spec["num_prompts"], spec["prompt_tokens"], spec["output_tokens"]
    shared = spec.get("shared_prefix_tokens", 0)
    prefix_ids = take_tokens(_filler(rng, shared * 2), shared, encode) if shared else []
    prompts = []
    for i in range(n):
        # Unique leading id in the suffix so prefix caching can only reuse the shared part.
        suffix = f" [req {i}] " + _filler(rng, (p_tok - shared) * 2)
        # Concatenate ids and decode once: decoding pieces separately and joining strings
        # merges tokens at the boundary and changes the length.
        ids = prefix_ids + take_tokens(suffix, p_tok - shared, encode)
        prompts.append({"id": i, "prompt": decode(ids), "max_tokens": o_tok})
    return prompts


def make_spider_prompts(spec: dict, seed: int, spider_dir: Path, tokenizer) -> list[dict]:
    """Real dev questions in the exact serving format (schema first, chat template), sampled
    with a fixed seed. Output length is natural (run the client with --no-ignore-eos) and
    max_tokens is only a cap. Dev has 20 databases, so many prompts share a schema prefix."""
    from evaluation.spider import db_path, schema_ddl
    from finetune.prompt import build_messages, render

    examples = json.loads((spider_dir / "dev.json").read_text())
    picked = random.Random(seed).sample(range(len(examples)), spec["num_prompts"])
    schemas: dict[str, str] = {}
    prompts = []
    for i, idx in enumerate(picked):
        ex = examples[idx]
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = schema_ddl(db_path(spider_dir, ex["db_id"]))
        prompt = render(tokenizer, build_messages(schemas[ex["db_id"]], ex["question"]),
                        add_generation_prompt=True)
        prompts.append({"id": i, "prompt": prompt, "max_tokens": spec["output_tokens"],
                        "dev_index": idx, "db_id": ex["db_id"]})
    return prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/workloads.yaml")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="data/workloads")
    p.add_argument("--spider-dir", type=Path, default=None, help="needed for spider_* workloads")
    a = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in cfg["workloads"].items():
        if spec.get("source") == "spider_dev":
            if a.spider_dir is None:
                print(f"skip {name}: pass --spider-dir")
                continue
            prompts = make_spider_prompts(spec, cfg["seed"], a.spider_dir, tok)
        else:
            prompts = make_prompts(
                spec, cfg["seed"],
                encode=lambda s: tok.encode(s, add_special_tokens=False),
                decode=lambda ids: tok.decode(ids),
            )
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in prompts))
        manifest[name] = {"num_prompts": len(prompts),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"wrote {len(prompts)} prompts -> {path}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
