"""Contract tests for cross-validation outputs (cv_results, cv_per_origin, best_params)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.splits import build_folds

ROOT = Path(__file__).resolve().parents[1]


def test_cv_results_contract():
    path = ROOT / "data" / "processed" / "cv_results.csv"
    assert path.exists(), f"{path} not found"

    df = pd.read_csv(path)
    # cv_results has 15 rows.
    assert len(df) == 15

    expected_cols = [
        "model",
        "params_id",
        "params_json",
        "wape",
        "mae",
        "rmse",
        "bias",
        "wape_1_7",
        "wape_8_14",
        "wape_15_28",
        "n_origins_better_than_baseline",
        "best_for_model",
    ]
    assert list(df.columns) == expected_cols
    # No duration column.
    assert not any("time" in c.lower() or "duration" in c.lower() or "elapsed" in c.lower() for c in df.columns)

    # Exactly 2 best_for_model rows (one for Prophet and one for XGBoost).
    assert df["best_for_model"].sum() == 2
    best_models = set(df.loc[df["best_for_model"], "model"])
    assert best_models == {"prophet", "xgboost"}


def test_cv_per_origin_contract():
    path = ROOT / "data" / "processed" / "cv_per_origin.csv"
    assert path.exists(), f"{path} not found"

    df = pd.read_csv(path)
    # cv_per_origin has 345 rows (15 candidates * 23 origins).
    assert len(df) == 345

    expected_cols = ["model", "params_id", "origin", "wape"]
    assert list(df.columns) == expected_cols

    config = load_config(ROOT / "configs" / "config.yaml")
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_revenue.csv", parse_dates=["date"])
    dev_folds = build_folds(daily, config, "dev")
    expected_dev_origins = sorted([pd.Timestamp(f["origin"]).strftime("%Y-%m-%d") for f in dev_folds])
    assert len(expected_dev_origins) == 23

    # Each params_id has 23 origins, all for development.
    unique_params = df["params_id"].unique()
    assert len(unique_params) == 15

    for pid in unique_params:
        sub = df[df["params_id"] == pid]
        assert len(sub) == 23
        sub_origins = sorted(sub["origin"].tolist())
        assert sub_origins == expected_dev_origins


def test_best_params_contract():
    path = ROOT / "configs" / "best_params.json"
    assert path.exists(), f"{path} not found"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # best_params.json contains a champion among baseline, prophet, and xgboost.
    assert "champion" in data
    assert data["champion"] in {"baseline", "prophet", "xgboost"}

    assert "rule" in data
    assert "rule_applied" in data
    assert "baseline" in data
    assert "prophet" in data
    assert "xgboost" in data

    # Verify required fields for the baseline and challengers.
    for key in ["prophet", "xgboost"]:
        assert "qualified" in data[key]
        assert "n_origins_better_than_baseline" in data[key]
        assert "wape" in data[key]
