"""Banc d'essai commun aux fonctions de prévision."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd


ForecastFunction = Callable[
    [pd.DataFrame, str | pd.Timestamp, int, Mapping[str, Any]], pd.DataFrame
]


def run_backtest(
    forecast_fn: ForecastFunction,
    daily: pd.DataFrame,
    folds: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Exécute une fonction de prévision et rattache les valeurs réelles."""
    required_daily = {"date", "revenue"}
    missing_daily = required_daily.difference(daily.columns)
    if missing_daily:
        raise ValueError(f"Colonnes manquantes dans daily: {sorted(missing_daily)}")
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    actual_by_date = work.drop_duplicates("date").set_index("date")["revenue"].astype(float)
    rows: list[pd.DataFrame] = []
    horizon = int(config["evaluation"]["horizon"])
    for fold in folds:
        prediction = forecast_fn(work, fold["origin"], horizon, config)
        expected_dates = pd.date_range(
            pd.Timestamp(fold["origin"]) + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        if not prediction["date"].equals(pd.Series(expected_dates, name="date")):
            predicted_dates = pd.DatetimeIndex(pd.to_datetime(prediction["date"]))
            if not predicted_dates.equals(expected_dates):
                raise ValueError(f"Dates prédites incorrectes pour {fold['origin']}")
        missing_actual = expected_dates.difference(actual_by_date.index)
        if len(missing_actual):
            raise ValueError(
                "Revenu réel manquant pour les dates cibles: "
                + ", ".join(str(value.date()) for value in missing_actual)
            )
        result = prediction[["date", "horizon", "prediction"]].copy()
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        result["actual"] = result["date"].map(actual_by_date)
        result.insert(0, "kind", fold["kind"])
        result.insert(1, "origin", pd.Timestamp(fold["origin"]).normalize())
        result["regime"] = fold["regime"]
        rows.append(result[[
            "kind", "origin", "date", "horizon", "regime", "actual", "prediction"
        ]])
    if not rows:
        return pd.DataFrame(
            columns=["kind", "origin", "date", "horizon", "regime", "actual", "prediction"]
        )
    return pd.concat(rows, ignore_index=True)
