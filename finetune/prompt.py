"""Prompt format shared by SFT, evaluation and the benchmark's spider workload.

Keeping one builder matters: a train/serve prompt mismatch silently costs accuracy, and
putting the schema first gives prefix caching something to reuse (same DB => same prefix).
"""

from __future__ import annotations

import re

SYSTEM = (
    "You are a text-to-SQL assistant. Given a SQLite schema and a question, reply with one "
    "SQLite query and nothing else."
)


def build_messages(schema: str, question: str, answer_sql: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    if answer_sql is not None:
        msgs.append({"role": "assistant", "content": answer_sql.strip()})
    return msgs


def render(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    # Qwen3 defaults to "thinking" mode; SQL generation doesn't need it and it would blow up
    # output length (and therefore every latency number).
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def render_generation_prompt(messages: list[dict]) -> str:
    """`render(tok, messages, add_generation_prompt=True)` without a tokenizer: Qwen3's ChatML
    with thinking off, for plain system/user/assistant text messages. The gateway uses it so its
    image needs no `transformers`; tests/test_prompt.py checks it byte-for-byte against the real
    template, so it can't drift from what SFT and the Week 3 evals used."""
    out = [f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages]
    return "".join(out) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_sql(text: str) -> str:
    text = _THINK.sub("", text)
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    return sql.strip().split(";")[0].strip()
