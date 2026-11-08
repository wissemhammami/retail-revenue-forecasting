"""Aggregate cleaned transactions into calendar series."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _full_dates(config: Mapping[str, Any]) -> pd.DatetimeIndex:
    """Retourne l'index complet jusqu'au dernier jour non partiel."""
    start = pd.Timestamp(config["expected"]["date_min"])
    partial = pd.Timestamp(config["aggregation"]["partial_last_day"])
    return pd.date_range(start, partial - pd.Timedelta(days=1), freq="D")


def _closed_dates(config: Mapping[str, Any]) -> set[pd.Timestamp]:
    """Return closure dates declared in the configuration."""
    return {
        pd.Timestamp(date)
        for date in config["aggregation"].get("closure_dates", [])
    }


def daily_revenue(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Aggregate revenue for each complete calendar day."""
    dates = _full_dates(config)
    work = df.copy()
    work["date"] = pd.to_datetime(work["InvoiceDate"]).dt.normalize()
    grouped = work.groupby("date").agg(
        revenue=("Revenue", "sum"),
        n_transactions=("Revenue", "size"),
    )
    result = grouped.reindex(dates, fill_value=0).rename_axis("date").reset_index()
    closed_dates = _closed_dates(config)
    result["is_trading_day"] = result["n_transactions"].gt(0)
    result.loc[
        result["n_transactions"].eq(0)
        & (
            result["date"].dt.dayofweek.eq(5)
            | result["date"].isin(closed_dates)
        ),
        "is_trading_day",
    ] = False
    assert (
        result.loc[result["n_transactions"].gt(0), "is_trading_day"].all()
    ), "A day with transactions cannot be classified as closed"
    result["is_partial_day"] = False
    return result[
        ["date", "revenue", "n_transactions", "is_trading_day", "is_partial_day"]
    ]


def weekly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily series and flag partial weeks."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    periods = work["date"].dt.to_period("W-SUN")
    result = work.assign(week=periods).groupby("week", as_index=False).agg(
        date=("date", "min"),
        revenue=("revenue", "sum"),
        n_transactions=("n_transactions", "sum"),
    )
    result["is_partial_week"] = (
        (result["date"] != result["date"].dt.to_period("W-SUN").dt.start_time)
        | (result["date"] == result["date"].max())
    )
    if not result.empty:
        result.loc[result.index[-1], "is_partial_week"] = True
    return result[["date", "revenue", "n_transactions", "is_partial_week"]]


def detect_anomalous_days(
    daily: pd.DataFrame, window: int = 28, k: float = 5
) -> pd.DataFrame:
    """Flag deviations greater than k times the rolling MAD."""
    work = daily.copy()
    values = work["revenue"].astype(float)
    rolling_median = values.rolling(window, min_periods=window).median()
    deviations = (values - rolling_median).abs()
    mad = deviations.rolling(window, min_periods=window).median()
    score = deviations / mad.replace(0, np.nan)
    score = score.mask(mad.eq(0) & deviations.gt(0), np.inf)
    mask = score.gt(k).fillna(False)
    return pd.DataFrame(
        {
            "date": work.loc[mask, "date"].values,
            "revenue": values.loc[mask].values,
            "rolling_median": rolling_median.loc[mask].values,
            "score": score.loc[mask].values,
        }
    )
