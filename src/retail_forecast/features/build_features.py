"""Construction de variables calendaires et retardées sans fuite temporelle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "week_of_year", "day_of_month",
    "is_trading_day", "is_pre_christmas", "is_post_christmas",
    "days_to_christmas", "has_full_history",
]


def _calendar_features(index: pd.DatetimeIndex, trading: pd.Series) -> pd.DataFrame:
    """Construit les variables connues à l'avance."""
    christmas = pd.to_datetime(index.year.astype(str) + "-12-24")
    days_to_christmas = (christmas - index).days
    result = pd.DataFrame(index=index)
    result["day_of_week"] = index.dayofweek
    result["month"] = index.month
    result["week_of_year"] = index.isocalendar().week.astype("int64").to_numpy()
    result["day_of_month"] = index.day
    result["is_trading_day"] = trading.reindex(index).fillna(False).astype(bool).to_numpy()
    result["is_pre_christmas"] = ((days_to_christmas >= 1) & (days_to_christmas <= 7))
    result["is_post_christmas"] = (
        ((index.month == 1) & (index.day >= 4) & (index.day <= 18))
    )
    result["days_to_christmas"] = days_to_christmas
    return result


def _feature_frame(revenue: pd.Series, trading: pd.Series) -> pd.DataFrame:
    """Construit les retards et moyennes avec un décalage d'un jour."""
    shifted = revenue.shift(1)
    result = pd.DataFrame(index=revenue.index)
    result["lag_7"] = revenue.shift(7)
    result["lag_14"] = revenue.shift(14)
    result["lag_28"] = revenue.shift(28)
    result["rolling_mean_7"] = shifted.rolling(7, min_periods=7).mean()
    result["rolling_mean_28"] = shifted.rolling(28, min_periods=28).mean()
    result = result.join(_calendar_features(revenue.index, trading))
    result["has_full_history"] = result[
        ["lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"]
    ].notna().all(axis=1)
    return result[FEATURE_COLUMNS]


def build_features(daily: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Construit les features historiques sur tous les jours calendaires."""
    del config
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date").drop_duplicates("date").set_index("date")
    revenue = work["revenue"].astype(float).where(work["is_trading_day"].astype(bool), 0.0)
    return _feature_frame(revenue, work["is_trading_day"])


def features_for_origin(
    daily: pd.DataFrame,
    origin: str | pd.Timestamp,
    horizon: int,
    predicted_values: Mapping[str | pd.Timestamp, float] | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Construit récursivement les features cibles avec revenus connus jusqu'à l'origine."""
    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif")
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date").set_index("date")
    origin = pd.Timestamp(origin).normalize()
    target_index = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    full_index = pd.date_range(work.index.min(), target_index[-1], freq="D")
    trading = work["is_trading_day"].reindex(full_index).fillna(False).astype(bool)
    known_revenue = work["revenue"].astype(float).where(
        work["is_trading_day"].astype(bool), 0.0
    )
    revenue = known_revenue.reindex(full_index)
    future_index = revenue.index > origin
    revenue.loc[future_index] = np.where(
        trading.loc[future_index].to_numpy(),
        np.nan,
        0.0,
    )
    if predicted_values is not None:
        if isinstance(predicted_values, Mapping):
            for key, value in predicted_values.items():
                timestamp = pd.Timestamp(key).normalize()
                if timestamp > origin and timestamp in revenue.index:
                    revenue.loc[timestamp] = float(value)
        else:
            if len(predicted_values) != horizon:
                raise ValueError("predicted_values doit contenir horizon valeurs")
            revenue.loc[target_index] = np.asarray(predicted_values, dtype=float)
    result = _feature_frame(revenue, trading)
    return result.loc[target_index]
