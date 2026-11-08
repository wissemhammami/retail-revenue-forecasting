"""Forecasting metrics and backtest aggregations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _arrays(actual: Iterable[float], pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert two vectors and verify their length and non-emptiness."""
    actual_array = np.asarray(list(actual), dtype=float)
    pred_array = np.asarray(list(pred), dtype=float)
    if actual_array.shape != pred_array.shape:
        raise ValueError("actual and pred must have the same length")
    if actual_array.size == 0:
        raise ValueError("actual and pred must not be empty")
    return actual_array, pred_array


def wape(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Return the weighted absolute error by actual revenue.

    If the absolute sum of actual values is zero, WAPE is mathematically
    undefined and raises ValueError rather than producing an artificial
    value. MAE, RMSE, and bias remain defined when all actual values are zero.
    """
    actual_array, pred_array = _arrays(actual, pred)
    denominator = float(np.abs(actual_array).sum())
    if denominator == 0:
        raise ValueError("WAPE undefined: the sum of actual values is zero")
    return float(np.abs(pred_array - actual_array).sum() / denominator)


def mae(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Return the mean absolute error."""
    actual_array, pred_array = _arrays(actual, pred)
    return float(np.abs(pred_array - actual_array).mean())


def rmse(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Return the root mean squared error."""
    actual_array, pred_array = _arrays(actual, pred)
    return float(np.sqrt(np.square(pred_array - actual_array).mean()))


def bias(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Return the mean bias, positive when over-forecasting."""
    actual_array, pred_array = _arrays(actual, pred)
    return float((pred_array - actual_array).mean())


def horizon_bucket(h: int, buckets: Iterable[Iterable[int]]) -> str:
    """Map a horizon to a configured bucket label."""
    horizon = int(h)
    for bucket in buckets:
        start, end = (int(value) for value in bucket)
        if start <= horizon <= end:
            return f"{start}-{end}"
    raise ValueError(f"Horizon outside configured buckets: {horizon}")


def summarize(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Aggregate metrics across all origin-date pairs in each group."""
    group_columns = [by] if isinstance(by, str) else (by or [])
    required = {"actual", "prediction", *group_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for summarize: {sorted(missing)}")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "wape": wape(group["actual"], group["prediction"]),
            "mae": mae(group["actual"], group["prediction"]),
            "rmse": rmse(group["actual"], group["prediction"]),
            "bias": bias(group["actual"], group["prediction"]),
        })

    if not group_columns:
        return aggregate(df).to_frame().T
    return df.groupby(group_columns, sort=False, dropna=False).apply(
        aggregate, include_groups=False
    ).reset_index()


def per_origin_wape(df: pd.DataFrame) -> pd.DataFrame:
    """Return WAPE for each origin."""
    if "origin" not in df.columns:
        raise ValueError("The origin column is required for per_origin_wape")
    return (
        df.groupby("origin", sort=True)
        .apply(lambda group: wape(group["actual"], group["prediction"]), include_groups=False)
        .rename("wape")
        .reset_index()
    )
