"""Vérifie l'invariance des prévisions à l'historique postérieur à l'origine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.models.baseline import naive_seasonal_fallback_forecast
from retail_forecast.models.prophet_model import forecast as prophet_forecast
from retail_forecast.models.xgb_model import forecast as xgb_forecast


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = pd.Timestamp("2011-06-07")
HORIZON = 28


def test_dev_forecasts_are_invariant_to_post_origin_rows() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_revenue.csv", parse_dates=["date"])
    daily_truncated = daily.loc[daily["date"] <= ORIGIN].copy()

    with (ROOT / "configs" / "best_params.json").open(encoding="utf-8") as handle:
        best_params = json.load(handle)

    forecasts = [
        (
            "baseline",
            lambda frame: naive_seasonal_fallback_forecast(
                frame, ORIGIN, HORIZON, config
            ),
            {},
        ),
        (
            "prophet",
            lambda frame: prophet_forecast(
                frame,
                ORIGIN,
                HORIZON,
                config,
                params=best_params["prophet"]["params"],
            ),
            best_params["prophet"]["params"],
        ),
        (
            "xgboost",
            lambda frame: xgb_forecast(
                frame,
                ORIGIN,
                HORIZON,
                config,
                params=best_params["xgboost"]["params"],
            ),
            best_params["xgboost"]["params"],
        ),
    ]

    for model_name, forecast_fn, _params in forecasts:
        full_prediction = forecast_fn(daily)
        truncated_prediction = forecast_fn(daily_truncated)
        np.testing.assert_allclose(
            full_prediction["prediction"].to_numpy(),
            truncated_prediction["prediction"].to_numpy(),
            err_msg=f"Prévision non invariante pour {model_name}",
        )
