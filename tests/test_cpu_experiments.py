import csv
from pathlib import Path

import yaml

from bench.cpu_experiments import build_command, client_command, expand, merge_summaries

PLAN = {
    "models": {"q4_k_m": "a.gguf", "q8_0": "b.gguf"},
    "baseline": {"threads": 4, "parallel": 4, "flash-attn": "on", "cache-prompt": True},
    "levels": {"spider_v1": [1, 4]},
    "experiments": {
        "q4_k_m": {"workloads": ["spider_v1"]},
        "no_cache": {"set": {"cache-prompt": False}, "workloads": ["spider_v1"]},
        "q8_0": {"model": "q8_0", "quality": "gguf_q8_0", "workloads": ["spider_v1"]},
    },
}


def test_expand_merges_flags_over_baseline():
    exps = {e["name"]: e for e in expand(PLAN)}
    assert exps["no_cache"]["flags"]["cache-prompt"] is False
    assert exps["no_cache"]["flags"]["threads"] == 4 and PLAN["baseline"]["cache-prompt"] is True
    assert exps["q4_k_m"]["model"] == "q4_k_m" and exps["q8_0"]["quality"] == "gguf_q8_0"


def test_build_command():
    cmd = build_command("llama-server", Path("m.gguf"),
                        {"threads": 2, "cache-prompt": False, "metrics": True, "flash-attn": "on"},
                        "qwen-sql", 8080)
    assert cmd[:7] == ["llama-server", "--model", "m.gguf", "--alias", "qwen-sql", "--port", "8080"]
    assert "--no-cache-prompt" in cmd and "--metrics" in cmd
    assert cmd[cmd.index("--flash-attn") + 1] == "on"


def test_client_command_marks_fresh_server():
    cmd = client_command("q4_k_m", "spider_v1", 4, "qwen-sql", Path("w"), Path("o"), 24, 3)
    assert "--fresh-server" in cmd and "--no-ignore-eos" in cmd
    assert cmd[cmd.index("--concurrency") + 1] == "4"


def test_merge_summaries(tmp_path):
    for c, tok in ((1, 10), (4, 30)):
        d = tmp_path / f"c{c}"
        d.mkdir()
        (d / "summary.csv").write_text(f"label,mode,output_tok_per_s\nx,c{c},{tok}\n")
    merge_summaries([tmp_path / "c1", tmp_path / "c4", tmp_path / "missing"], tmp_path / "summary.csv")
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [r["mode"] for r in rows] == ["c1", "c4"]


def test_real_cpu_plan_is_valid():
    plan = yaml.safe_load(Path("configs/cpu_experiments.yaml").read_text())
    workloads = yaml.safe_load(Path("configs/workloads.yaml").read_text())["workloads"]
    exps = expand(plan)
    assert exps[0]["name"] == "q4_k_m"
    for e in exps:
        assert e["model"] in plan["models"] and set(e["workloads"]) <= set(workloads), e["name"]
        # YAML 1.1 reads bare on/off as booleans; llama.cpp's --flash-attn wants the words
        assert e["flags"]["flash-attn"] in ("on", "off", "auto"), e["name"]
    assert set(plan["slo"]) >= {wl for e in exps for wl in e["workloads"]}
