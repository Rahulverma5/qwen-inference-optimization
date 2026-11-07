from bench.workloads import make_prompts


def encode(s):  # whitespace "tokenizer" for tests
    return s.split()


def decode(ids):
    return " ".join(ids)


def test_shared_prefix_and_lengths():
    spec = {"num_prompts": 5, "prompt_tokens": 50, "output_tokens": 8, "shared_prefix_tokens": 30}
    ps = make_prompts(spec, seed=1, encode=encode, decode=decode)
    prefix = " ".join(ps[0]["prompt"].split()[:30])
    assert all(p["prompt"].startswith(prefix) for p in ps)
    assert all(len(encode(p["prompt"])) == 50 for p in ps)
    assert len({p["prompt"] for p in ps}) == 5


def test_deterministic():
    spec = {"num_prompts": 3, "prompt_tokens": 20, "output_tokens": 4}
    assert make_prompts(spec, 7, encode, decode) == make_prompts(spec, 7, encode, decode)


def test_spider_prompts_use_serving_format(tmp_path):
    import json
    import sqlite3

    from bench.workloads import make_spider_prompts

    db = tmp_path / "database" / "pets" / "pets.sqlite"
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).execute("CREATE TABLE pet (id INTEGER, PetType TEXT)").connection.commit()
    exs = [{"db_id": "pets", "question": f"q{i}"} for i in range(10)]
    (tmp_path / "dev.json").write_text(json.dumps(exs))

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
            return "|".join(m["content"] for m in msgs)

    ps = make_spider_prompts({"num_prompts": 4, "output_tokens": 128}, 0, tmp_path, Tok())
    assert len(ps) == 4 and len({p["dev_index"] for p in ps}) == 4
    assert all("PetType" in p["prompt"] and p["max_tokens"] == 128 for p in ps)
    assert ps == make_spider_prompts({"num_prompts": 4, "output_tokens": 128}, 0, tmp_path, Tok())


def test_mixed_lengths_without_changing_plain_workloads():
    plain = {"num_prompts": 50, "prompt_tokens": 10, "output_tokens": 4}
    mixed = plain | {"long_prompt_tokens": 80, "long_fraction": 0.2}
    lengths = [len(encode(p["prompt"])) for p in make_prompts(mixed, 3, encode, decode)]
    assert set(lengths) == {10, 80} and 3 <= lengths.count(80) <= 20
    # A spec without the long keys draws no extra random numbers: Week 2 files stay byte-identical.
    assert make_prompts(plain, 3, encode, decode) == make_prompts(
        plain | {"long_fraction": 0.5}, 3, encode, decode)
