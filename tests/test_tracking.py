import sys
import types

import tracking


def test_noop_without_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.log_run("e", "r", {"a": 1}, {"m": 1.0}) is None


def test_drops_nan_and_non_numeric_metrics(monkeypatch):
    logged = {}

    class Run:
        info = types.SimpleNamespace(run_id="abc")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake = types.SimpleNamespace(
        set_experiment=lambda name: None,
        start_run=lambda **kw: Run(),
        log_params=lambda p: logged.update(params=p),
        log_metrics=lambda m: logged.update(metrics=m),
        log_artifact=lambda p: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/x")
    rid = tracking.log_run("e", "r", {"tp": 2}, {"ok": 1, "nan": float("nan"), "s": "x"})
    assert rid == "abc"
    assert logged["metrics"] == {"ok": 1.0}
    assert logged["params"] == {"tp": "2"}
