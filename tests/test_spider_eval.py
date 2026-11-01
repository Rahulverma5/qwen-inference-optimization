import sqlite3

import pytest

from evaluation.spider import exec_match, results_match, schema_ddl, score


@pytest.fixture
def spider_dir(tmp_path):
    d = tmp_path / "database" / "shop"
    d.mkdir(parents=True)
    with sqlite3.connect(d / "shop.sqlite") as c:
        c.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
        c.executemany("INSERT INTO item VALUES (?,?,?)", [(1, "a", 3.0), (2, "b", 1.0), (3, "c", 2.0)])
    return tmp_path


def test_schema_ddl(spider_dir):
    assert "CREATE TABLE item" in schema_ddl(spider_dir / "database/shop/shop.sqlite")


def test_unordered_vs_ordered_comparison():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)  # multiset, not set


def test_exec_match(spider_dir):
    db = spider_dir / "database/shop/shop.sqlite"
    gold = "SELECT name FROM item ORDER BY price"
    assert exec_match(db, "SELECT name FROM item ORDER BY price ASC", gold) == (True, True)
    assert exec_match(db, "SELECT name FROM item", gold) == (True, False)  # wrong order
    assert exec_match(db, "SELEC name", gold) == (False, False)


def test_read_only(spider_dir):
    db = spider_dir / "database/shop/shop.sqlite"
    assert exec_match(db, "DELETE FROM item", "SELECT 1") == (False, False)
    assert exec_match(db, "SELECT count(*) FROM item", "SELECT 3") == (True, True)


def test_runaway_query_times_out(spider_dir, monkeypatch):
    monkeypatch.setattr("evaluation.spider.QUERY_TIMEOUT_S", 0.2)
    from evaluation import spider
    db = spider_dir / "database/shop/shop.sqlite"
    slow = ("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) "
            "SELECT count(*) FROM r")
    with pytest.raises(sqlite3.OperationalError):
        spider.execute(db, slow, timeout_s=0.2)


def test_score(spider_dir):
    examples = [{"db_id": "shop", "query": "SELECT count(*) FROM item"},
                {"db_id": "shop", "query": "SELECT max(price) FROM item"}]
    s = score(spider_dir, examples, {0: "SELECT count(id) FROM item", 1: "garbage"})
    assert s == {"n": 2, "exec_accuracy": 0.5, "valid_sql_rate": 0.5}
