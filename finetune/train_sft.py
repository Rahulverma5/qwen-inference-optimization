"""QLoRA SFT of Qwen3 on Spider train, sized for a Kaggle T4 (16 GB, fp16, no bf16).

Single GPU:   python -m finetune.train_sft --spider-dir data/spider_data --out models/sft
Both T4s:     accelerate launch --num_processes 2 -m finetune.train_sft ...   (DDP)

Loss is on the completion (SQL) only: the dataset is prompt/completion pairs rendered with
the same chat template used at inference, so train and serve prompts are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluation.spider import db_path, schema_ddl
from finetune.prompt import build_messages, render

TRAIN_SPLITS = ("train_spider.json", "train_others.json")  # standard 8,659-example Spider train


def build_rows(spider_dir: Path, splits, tokenizer, max_len: int,
               limit: int | None = None) -> tuple[list[dict], int]:
    """Prompt/completion pairs; drops examples longer than max_len instead of truncating them.

    TRL truncates from the right, and the SQL completion sits at the end, so truncation would
    train on a cut-off answer (or on no answer at all). ~1% of Spider train exceeds 2,048 tokens.
    """
    examples = [ex for s in splits for ex in json.loads((spider_dir / s).read_text())][:limit]
    rows, dropped = [], 0
    for ex in examples:
        msgs = build_messages(schema_ddl(db_path(spider_dir, ex["db_id"])), ex["question"])
        row = {
            "prompt": render(tokenizer, msgs, add_generation_prompt=True),
            "completion": ex["query"].strip() + tokenizer.eos_token,
        }
        n_tok = len(tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False))
        if n_tok > max_len:
            dropped += 1
            continue
        rows.append(row)
    return rows, dropped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--spider-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("models/sft"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None, help="subset for smoke tests")
    p.add_argument("--report-to", default="mlflow" if os.environ.get("MLFLOW_TRACKING_URI") else "none",
                   help="mlflow | none (defaults to mlflow when MLFLOW_TRACKING_URI is set)")
    p.add_argument("--run-name", default=None, help="MLflow run name, e.g. r32-lr2e-4")
    a = p.parse_args()

    if a.report_to == "mlflow":
        # Read by the HF Trainer's MLflowCallback / MLflow itself at run start.
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "qlora-sft")
        os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows, dropped = build_rows(a.spider_dir, TRAIN_SPLITS, tok, a.max_len, a.limit)
    print(f"train examples: {len(rows)} kept, {dropped} dropped (> {a.max_len} tokens)")
    from datasets import Dataset

    train = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4: no bf16
        ),
        torch_dtype=torch.float16,
        attn_implementation="sdpa",  # FlashAttention-2 needs sm80+
        # 4-bit weights can't be moved after loading: put each DDP process's copy on its own GPU.
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=str(a.out),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required with DDP + PEFT
        ddp_find_unused_parameters=False,
        group_by_length=True,  # prompts range 100-2,000 tokens; batching similar lengths cuts padding
        fp16=True,
        max_length=a.max_len,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,  # Kaggle sessions die at 12 h; checkpoints let you resume
        save_total_limit=2,
        report_to=a.report_to,
        run_name=a.run_name,
        seed=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok,
                         peft_config=peft_cfg)
    trainer.train(resume_from_checkpoint=any(a.out.glob("checkpoint-*")))
    trainer.save_model(str(a.out / "adapter"))
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(a.out / "trainer_state.json"))  # loss curve for the report


if __name__ == "__main__":
    main()
