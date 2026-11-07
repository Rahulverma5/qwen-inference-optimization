import json
import sqlite3
import sys
from pathlib import Path

from finetune.train_sft import build_rows
from quantize.calibration import calibration_texts, sample_examples, tokenize
from quantize.gguf import export_commands, gguf_name, setup_commands


class FakeTok:
    """Whitespace tokenizer with a Qwen3-like chat template (as in test_train_rows)."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return " ".join(m["content"] for m in messages) + (" <assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_sample_is_deterministic_without_replacement():
    xs = list(range(100))
    s = sample_examples(xs, 10, seed=0)
    assert s == sample_examples(xs, 10, seed=0) and len(set(s)) == 10
    assert s != sample_examples(xs, 10, seed=1)
    assert sample_examples(xs, 500, seed=0) == xs


def test_calibration_text_is_the_sft_row(tmp_path):
    """Calibration must see exactly what SFT trained on: prompt + gold SQL + EOS."""
    db = tmp_path / "database" / "s"
    db.mkdir(parents=True)
    with sqlite3.connect(db / "s.sqlite") as c:
        c.execute("CREATE TABLE t (a INT)")
    exs = [{"db_id": "s", "question": f"q{i}", "query": f" SELECT {i} FROM t "} for i in range(5)]
    (tmp_path / "train.json").write_text(json.dumps(exs))
    texts = calibration_texts(tmp_path, ["train.json"], FakeTok(), n=5, seed=0)
    rows, _ = build_rows(tmp_path, ("train.json",), FakeTok(), max_len=10_000)
    assert texts == [r["prompt"] + r["completion"] for r in rows]


def test_tokenize_truncates_without_padding():
    out = tokenize(["a b c d", "e"], FakeTok(), max_len=3)
    assert out["input_ids"] == [["a", "b", "c"], ["e"]]
    assert out["attention_mask"] == [[1, 1, 1], [1]]


def test_gguf_commands():
    llama, model, out = Path("ll"), Path("m"), Path("o")
    convert, q8, q4 = export_commands(llama, model, out, "qwen", ["Q8_0", "Q4_K_M"])
    assert convert[0] == sys.executable and convert[-1] == str(out / gguf_name("qwen", "f16"))
    assert q4 == ["ll/build/bin/llama-quantize", "o/qwen-f16.gguf", "o/qwen-Q4_K_M.gguf", "Q4_K_M"]
    clone, configure, build = setup_commands("url", "b1", llama, 4)
    assert clone[clone.index("--branch") + 1] == "b1" and "-DGGML_CUDA=OFF" in configure
    assert {"llama-quantize", "llama-server"} <= set(build)
