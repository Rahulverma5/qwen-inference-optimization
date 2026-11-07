import sqlite3

import pytest

from evaluation.compare import compare, complexity


def test_complexity_buckets():
    assert complexity("SELECT count(*) FROM t") == "simple"
    assert complexity("SELECT a FROM t GROUP BY a") == "agg/order"
    assert complexity("SELECT a FROM t JOIN u ON t.id = u.id") == "join"
    assert complexity("SELECT a FROM t JOIN u ON t.id = u.id ORDER BY a") == "join+agg/order"
    assert complexity("SELECT a FROM t WHERE a IN (SELECT a FROM u)") == "nested/set-op"
    assert complexity("SELECT a FROM t EXCEPT SELECT a FROM u") == "nested/set-op"


@pytest.fixture
def spider_dir(tmp_path):
    d = tmp_path / "database" / "s"
    d.mkdir(parents=True)
    with sqlite3.connect(d / "s.sqlite") as c:
        c.execute("CREATE TABLE t (a INT)")
        c.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    return tmp_path


def test_flip_matrix_and_errors(spider_dir):
    examples = [{"db_id": "s", "query": "SELECT count(*) FROM t"},
                {"db_id": "s", "query": "SELECT max(a) FROM t"},
                {"db_id": "s", "query": "SELECT min(a) FROM t"}]
    a = {0: "SELECT count(*) FROM t", 1: "SELECT nope FROM t", 2: "SELECT min(a) FROM t"}
    b = {0: "SELECT count(a) FROM t", 1: "SELECT max(a) FROM t", 2: "SELECT 5"}
    r = compare(spider_dir, examples, a, b)
    assert (r["both_right"], r["fixed_by_b"], r["broken_by_b"], r["both_wrong"]) == (1, 1, 1, 0)
    assert r["exec_errors_a"] == {"no such column": 1}
    assert r["examples_fixed"] == [1] and r["examples_broken"] == [2]


def test_mcnemar_exact():
    from evaluation.compare import mcnemar_p

    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(3, 0) == pytest.approx(0.25)  # 2 * (1/2)^3
    assert mcnemar_p(160, 70) < 1e-8  # Week 1 SFT vs base: chi-squared gave ~4e-9
