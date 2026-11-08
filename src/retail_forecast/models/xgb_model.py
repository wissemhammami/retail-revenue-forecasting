"""XGBoost forecasting model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from retail_forecast.evaluation.closures import is_known_closed
from retail_forecast.features.build_features import build_features, features_for_origin

FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "week_of_year",
    "day_of_month",
    "is_pre_christmas",
    "is_post_christmas",
    "days_to_christmas",
]


def forecast(
    daily: pd.DataFrame,
    origin: str | pd.Timestamp,
    horizon: int,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Recursively forecast daily revenue with XGBoost."""
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

    # Features built on the series truncated at date <= origin.
    feats = build_features(work, config)

    # Training rows = trading days with all non-NaN features.
    train_mask = (
        feats["is_trading_day"].astype(bool)
        & feats[FEATURE_COLUMNS].notna().all(axis=1)
    )
    train_df = feats.loc[train_mask]

    X_train = train_df[FEATURE_COLUMNS]
    revenue_series = work.set_index("date")["revenue"].astype(float)
    y_train = revenue_series.loc[train_df.index]

    # Hyperparameters.
    xgb_cfg = config.get("models", {}).get("xgboost", {})
    fixed_params = dict(xgb_cfg.get("fixed", {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    }))
    default_params = dict(xgb_cfg.get("default", {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
    }))
    if params:
        default_params.update(params)

    model_params = {**fixed_params, **default_params}
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)

    target_dates = pd.date_range(
        origin_timestamp + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    horizons = np.arange(1, horizon + 1)
    closed_flags = is_known_closed(target_dates, config).to_numpy()

    predicted_values: dict[pd.Timestamp, float] = {}
    predictions: list[float] = []

    for t_date, is_closed in zip(target_dates, closed_flags):
        if is_closed:
            pred_val = 0.0
        else:
            step_feats = features_for_origin(
                work, origin_timestamp, horizon, predicted_values=predicted_values
            )
            x_step = step_feats.loc[[t_date], FEATURE_COLUMNS]
            raw_pred = float(model.predict(x_step)[0])
            pred_val = max(0.0, raw_pred)

        predicted_values[t_date] = pred_val
        predictions.append(pred_val)

    return pd.DataFrame({
        "date": target_dates,
        "horizon": horizons,
        "prediction": predictions,
    })
