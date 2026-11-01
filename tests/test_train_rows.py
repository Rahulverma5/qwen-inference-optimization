import json
import sqlite3

from finetune.train_sft import build_rows


class FakeTok:
    """Whitespace tokenizer with a chat template that mimics the Qwen3 layout."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        text = " ".join(m["content"] for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_drops_long_examples_instead_of_truncating(tmp_path):
    db = tmp_path / "database" / "s"
    db.mkdir(parents=True)
    with sqlite3.connect(db / "s.sqlite") as c:
        c.execute("CREATE TABLE t (a INT)")
    short = {"db_id": "s", "question": "how many", "query": "SELECT count(*) FROM t"}
    long = {"db_id": "s", "question": "word " * 500, "query": "SELECT a FROM t"}
    (tmp_path / "a.json").write_text(json.dumps([short, long]))
    (tmp_path / "b.json").write_text(json.dumps([short]))

    rows, dropped = build_rows(tmp_path, ("a.json", "b.json"), FakeTok(), max_len=100)
    assert dropped == 1 and len(rows) == 2
    assert rows[0]["completion"] == "SELECT count(*) FROM t<|im_end|>"
    assert rows[0]["prompt"].endswith("<assistant>")
