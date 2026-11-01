"""Experiment tracking with MLflow (the only tracker in this project).

  training    - finetune.train_sft uses the HF Trainer's MLflow callback (report_to="mlflow"):
                loss, LR, grad norm per step; GPU memory/util via MLflow system metrics.
  eval/bench  - log_run() below: one run per evaluation or benchmark config.
  release     - the model registry alias `champion` decides which variant ships.

Everything is a no-op unless MLFLOW_TRACKING_URI is set, so local tests never need a server.
Set on Kaggle / CI (a free hosted server such as DagsHub works from both):
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""

from __future__ import annotations

import math
import os
from pathlib import Path


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def log_run(experiment: str, run_name: str, params: dict, metrics: dict,
            artifacts: list[Path] = (), tags: dict | None = None) -> str | None:
    """Log one finished run; returns the MLflow run id, or None when tracking is off."""
    if not enabled():
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        # MLflow rejects NaN; drop metrics that weren't measurable (e.g. TPOT for 1-token outputs)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if isinstance(v, int | float) and not math.isnan(v)})
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
