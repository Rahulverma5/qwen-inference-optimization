import math

import pytest

from bench.metrics import RequestRecord, summarize


def rec(i, t_send, chunks, out_tokens, err=None):
    return RequestRecord(i, t_send, chunks, prompt_tokens=10, output_tokens=out_tokens, error=err)


def test_request_level_metrics():
    r = rec(0, 0.0, [0.5, 0.6, 0.8, 1.0], out_tokens=4)
    assert r.ttft == pytest.approx(0.5)
    assert r.e2e == pytest.approx(1.0)
    assert r.itls == pytest.approx([0.1, 0.2, 0.2])
    assert r.tpot == pytest.approx(0.5 / 3)


def test_failed_and_empty_requests_excluded():
    records = [
        rec(0, 0.0, [0.2, 0.4], out_tokens=2),
        rec(1, 0.0, [], out_tokens=0),                  # stream closed without tokens
        rec(2, 0.0, [], out_tokens=0, err="HTTP 429"),
    ]
    s = summarize(records, wall_time_s=2.0)
    assert s["num_ok"] == 1
    assert s["error_rate"] == pytest.approx(2 / 3)
    assert s["request_throughput_rps"] == pytest.approx(0.5)
    assert s["output_tok_per_s"] == pytest.approx(1.0)
    assert s["total_tok_per_s"] == pytest.approx(6.0)
    assert s["ttft_s_p50"] == pytest.approx(0.2)


def test_single_token_output_has_no_tpot():
    s = summarize([rec(0, 0.0, [0.3], out_tokens=1)], wall_time_s=1.0)
    assert math.isnan(s["tpot_s_p95"])
