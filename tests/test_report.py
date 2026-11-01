import pytest

from bench.report import capacity_table, knee, littles_ratio, max_under_slo


def row(c, tok_s, e2e, ttft, tpot, err=0.0, label="b/short_v1"):
    return {"label": label, "mode": f"c{c}", "concurrency": c, "output_tok_per_s": tok_s,
            "request_throughput_rps": tok_s / 128, "e2e_s_mean": e2e, "e2e_s_p95": e2e,
            "ttft_s_p95": ttft, "tpot_s_p95": tpot, "error_rate": err}


SWEEP = [row(1, 80, 1.6, 0.05, 0.012), row(4, 300, 1.7, 0.08, 0.013),
         row(16, 900, 2.3, 0.3, 0.017), row(64, 1000, 8.2, 2.5, 0.06)]


def test_knee_maximizes_power():
    assert knee(SWEEP)["concurrency"] == 16


def test_max_under_slo_stops_at_first_failure():
    slo = {"ttft_s_p95": 0.5, "tpot_s_p95": 0.05}
    assert max_under_slo(SWEEP, slo)["concurrency"] == 16
    lucky = [SWEEP[0], row(4, 300, 1.7, 0.9, 0.013), SWEEP[2]]  # c=4 fails, c=16 passes
    assert max_under_slo(lucky, slo)["concurrency"] == 1
    assert max_under_slo([row(1, 80, 1.6, 0.05, 0.012, err=0.01)], slo) is None


def test_littles_law_ratio():
    # 16 in flight, 900/128 req/s, 2.3 s each -> L = 7.03 * 2.3 = 16.2
    assert littles_ratio(SWEEP[2]) == pytest.approx(900 / 128 * 2.3 / 16)


def test_capacity_table_uses_workload_slo():
    t = capacity_table(SWEEP, {"short_v1": {"ttft_s_p95": 0.5}})[0]
    assert (t["knee_c"], t["slo_max_c"], t["peak_out_tok_s"]) == (16, 16, 1000)
