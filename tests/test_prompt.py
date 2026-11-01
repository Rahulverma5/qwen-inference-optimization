from finetune.prompt import build_messages, extract_sql


def test_extract_sql_variants():
    assert extract_sql("SELECT 1;") == "SELECT 1"
    assert extract_sql("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"
    assert extract_sql("<think>hmm</think>\nSELECT b FROM t; -- extra") == "SELECT b FROM t"


def test_schema_comes_first_for_prefix_caching():
    m = build_messages("CREATE TABLE t(a);", "how many?")
    assert m[1]["content"].startswith("Schema:\nCREATE TABLE t(a);")
    assert len(build_messages("s", "q", "SELECT 1")) == 3
