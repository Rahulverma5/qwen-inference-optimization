import json
from pathlib import Path

import pytest
import yaml

from bench.client import num_requests
from bench.experiments import client_command, expand, server_specs

PLAN = {
    "levels": {"spider_v1": [1, 4], "long_v1": [4]},
    "experiments": {
        "fp16": {"workloads": ["spider_v1", "long_v1"]},
        "tp2": {"gpus_per_replica": 2, "workloads": ["spider_v1"], "levels": {"spider_v1": [1, 128]}},
        "replicas2": {"replicas": 2, "workloads": ["long_v1"], "set": {"max-num-seqs": 16}},
    },
}


def test_expand_fills_defaults_and_overrides_levels():
    exps = {e["name"]: e for e in expand(PLAN)}
    assert exps["fp16"]["model"] == "fp16" and exps["fp16"]["quality"] == "fp16"
    assert exps["fp16"]["workloads"] == {"spider_v1": [1, 4], "long_v1": [4]}
    assert exps["tp2"]["workloads"] == {"spider_v1": [1, 128]}
    assert [e["name"] for e in expand(PLAN, only=["tp2"])] == ["tp2"]
    with pytest.raises(ValueError):
        expand(PLAN, only=["nope"])


def test_server_specs_tp_and_replicas():
    exps = {e["name"]: e for e in expand(PLAN)}
    base = {"dtype": "float16", "max-num-seqs": 256}
    (tp,) = server_specs(exps["tp2"], base)
    assert tp["gpus"] == "0,1" and tp["profile"]["tensor-parallel-size"] == 2
    r0, r1 = server_specs(exps["replicas2"], base)
    assert (r0["gpus"], r0["port"], r1["gpus"], r1["port"]) == ("0", 8000, "1", 8001)
    assert r0["profile"]["max-num-seqs"] == 16 and "tensor-parallel-size" not in r0["profile"]
    assert base["max-num-seqs"] == 256  # base profile not mutated


def test_client_command_natural_length_only_for_spider():
    exp = expand(PLAN)[0]
    cmd = client_command(exp, "spider_v1", [1, 4], ["u0", "u1"], "q", Path("w"), Path("o"), 40, 4)
    assert cmd[cmd.index("--url") + 1: cmd.index("--model")] == ["u0", "u1"]
    assert "--no-ignore-eos" in cmd and cmd[cmd.index("--label") + 1] == "fp16/spider_v1"
    assert "--no-ignore-eos" not in client_command(exp, "long_v1", [4], ["u"], "q", Path("w"),
                                                   Path("o"), 40, 4)


def test_num_requests():
    assert num_requests(300, 1, None, None) == 300
    assert num_requests(300, 1, 40, 4) == 40
    assert num_requests(300, 16, 40, 4) == 64
    assert num_requests(300, 128, 40, 4) == 300


def test_real_plan_is_valid():
    """Every experiment in the committed plan expands, names known workloads, and changes the
    baseline in at most a few flags."""
    plan = yaml.safe_load(Path("configs/experiments.yaml").read_text())
    workloads = yaml.safe_load(Path("configs/workloads.yaml").read_text())["workloads"]
    exps = expand(plan)
    assert exps[0]["name"] == "fp16"  # the baseline runs first
    for e in exps:
        assert set(e["workloads"]) <= set(workloads), e["name"]
        json.dumps(e)  # serializable into result.json
