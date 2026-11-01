import math

import pytest

from bench.server import build_command, parse_kv_capacity, parse_overrides
from bench.server_metrics import parse_prometheus, pick, summarize_samples

PROM = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen-sql"} 7.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen-sql"} 0.42
vllm:num_preemptions_total{engine="0",model_name="qwen-sql"} 3.0
vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="qwen-sql"} 12.0
"""


def test_parse_prometheus_and_name_fallback():
    m = parse_prometheus(PROM)
    assert m["vllm:num_requests_running"] == 7.0
    assert pick(m, ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")) == 0.42
    assert pick(m, ("vllm:missing",)) is None


def test_summarize_samples_peaks_and_counter_deltas():
    rows = [
        {"running": 1, "waiting": 0, "kv_usage": 0.1, "gpu_mem_mib": 9000, "preemptions": 5,
         "prefix_queries": 100, "prefix_hits": 10},
        {"running": 8, "waiting": 3, "kv_usage": 0.9, "gpu_mem_mib": 9100, "preemptions": 7,
         "prefix_queries": 1100, "prefix_hits": 760},
    ]
    s = summarize_samples(rows)
    assert (s["running_max"], s["waiting_max"], s["kv_usage_max"]) == (8, 3, 0.9)
    assert s["preemptions_delta"] == 2
    assert s["prefix_hit_rate"] == pytest.approx(0.75)


def test_summarize_samples_without_metrics():
    s = summarize_samples([{"t_s": 0.0, "gpu_mem_mib": None}])
    assert math.isnan(s["kv_usage_max"]) and math.isnan(s["prefix_hit_rate"])


def test_parse_kv_capacity():
    log = ("INFO ... GPU KV cache size: 81,456 tokens\n"
           "INFO ... Maximum concurrency for 4,096 tokens per request: 19.89x\n")
    assert parse_kv_capacity(log) == {"kv_cache_tokens": 81456, "max_concurrency_seq_len": 4096,
                                      "max_concurrency": 19.89}


def test_build_command_booleans_and_overrides():
    profile = {"dtype": "float16", "enable-prefix-caching": True} | parse_overrides(
        ["enable-prefix-caching=false", "max-num-seqs=64"])
    cmd = build_command("m", profile, "qwen-sql", 8000)
    assert "--no-enable-prefix-caching" in cmd and "--enable-prefix-caching" not in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "64"
