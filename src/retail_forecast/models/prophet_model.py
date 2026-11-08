"""Prophet forecasting model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
from prophet import Prophet

from retail_forecast.evaluation.closures import is_known_closed

# Reduce cmdstanpy and Prophet logs.
cmdstanpy.utils.get_logger().setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def forecast(
    daily: pd.DataFrame,
    origin: str | pd.Timestamp,
    horizon: int,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Forecast daily revenue with Prophet."""
    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    origin_timestamp = pd.Timestamp(origin).normalize()
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.loc[work["date"] <= origin_timestamp].sort_values("date")

    min_train_days = int(config.get("evaluation", {}).get("min_train_days", 90))
    unique_dates = work["date"].drop_duplicates()
    if len(unique_dates) < min_train_days:
        raise ValueError(
            f"Insufficient history for origin {origin_timestamp.date()}: "
            f"{len(unique_dates)} days available, minimum required {min_train_days}"
        )

    # Train on is_trading_day == True days from this history.
    open_days = work.loc[work["is_trading_day"].astype(bool)].copy()
    df_train = open_days[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
    df_train["y"] = df_train["y"].astype(float)

    # Parameters.
    prophet_cfg = config.get("models", {}).get("prophet", {})
    default_params = dict(prophet_cfg.get("default", {}))
    if params:
        default_params.update(params)

    changepoint_prior_scale = float(default_params.get("changepoint_prior_scale", 0.05))
    seasonality_mode = str(default_params.get("seasonality_mode", "additive"))

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    model.fit(df_train)

    target_dates = pd.date_range(
        origin_timestamp + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    horizons = np.arange(1, horizon + 1)

    future_df = pd.DataFrame({"ds": target_dates})
    forecast_df = model.predict(future_df)

    raw_preds = forecast_df["yhat"].to_numpy(dtype=float)
    predictions = np.maximum(0.0, raw_preds)

    # Known closed target days are forced to 0 via is_known_closed.
    closed_flags = is_known_closed(target_dates, config).to_numpy()
    predictions[closed_flags] = 0.0

    return pd.DataFrame({
        "date": target_dates,
        "horizon": horizons,
        "prediction": predictions,
    })
