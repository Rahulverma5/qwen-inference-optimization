"""Generate Spider predictions with vLLM offline inference (Kaggle GPU).

    python -m evaluation.generate --model Qwen/Qwen3-1.7B --spider-dir data/spider \
        --out reports/preds/base_fp16.jsonl [--quantization awq] [--limit 200]

The model arg can be a hub id or a local dir (merged SFT model, llm-compressor output).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.spider import db_path, schema_ddl, score
from finetune.prompt import build_messages, extract_sql, render
from tracking import log_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--split", default="dev.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--quantization", default=None)
    a = p.parse_args()

    from vllm import LLM, SamplingParams

    examples = json.loads((a.spider_dir / a.split).read_text())[: a.limit]
    llm = LLM(model=a.model, dtype="float16", max_model_len=a.max_model_len,
              tensor_parallel_size=a.tp, quantization=a.quantization, seed=0)
    tok = llm.get_tokenizer()
    prompts = [
        render(tok, build_messages(schema_ddl(db_path(a.spider_dir, ex["db_id"])), ex["question"]),
               add_generation_prompt=True)
        for ex in examples
    ]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
    elapsed = time.perf_counter() - t0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    preds = {}
    with a.out.open("w") as f:
        for i, o in enumerate(outs):
            preds[i] = extract_sql(o.outputs[0].text)
            f.write(json.dumps({"id": i, "sql": preds[i], "raw": o.outputs[0].text}) + "\n")
    result = score(a.spider_dir, examples, preds) | {"model": a.model, "gen_time_s": elapsed}
    a.out.with_suffix(".score.json").write_text(json.dumps(result, indent=2))
    log_run("spider-eval", a.out.stem,
            params={"model": a.model, "quantization": a.quantization, "tp": a.tp,
                    "split": a.split, "limit": a.limit},
            metrics={k: result[k] for k in ("exec_accuracy", "valid_sql_rate", "gen_time_s")},
            artifacts=[a.out, a.out.with_suffix(".score.json")])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
