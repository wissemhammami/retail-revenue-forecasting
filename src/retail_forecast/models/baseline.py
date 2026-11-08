"""Modèles de référence sans apprentissage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.evaluation.closures import is_known_closed


def naive_seasonal_forecast(
    daily: pd.DataFrame,
    origin: str | pd.Timestamp,
    horizon: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Prévoit chaque jour par la valeur à la semaine saisonnière précédente."""
    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif")
    origin_timestamp = pd.Timestamp(origin).normalize()
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.loc[work["date"] <= origin_timestamp].sort_values("date")
    if work.empty:
        raise ValueError("Aucune donnée disponible jusqu'à l'origine")
    revenue_by_date = work.drop_duplicates("date").set_index("date")["revenue"].astype(float)
    target_dates = pd.date_range(
        origin_timestamp + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    horizons = np.arange(1, horizon + 1)
    k_values = np.ceil(horizons / 7).astype(int)
    source_dates = target_dates - pd.to_timedelta(7 * k_values, unit="D")
    if source_dates.min() < revenue_by_date.index.min():
        raise ValueError(
            "La date source du naïf tombe avant le début de la série: "
            f"{source_dates.min().date()}"
        )
    missing_sources = source_dates.difference(revenue_by_date.index)
    if len(missing_sources):
        raise ValueError(
            "Dates sources manquantes pour le naïf: "
            + ", ".join(str(value.date()) for value in missing_sources)
        )
    predictions = revenue_by_date.loc[source_dates].to_numpy(dtype=float, copy=True)
    closed = is_known_closed(target_dates, config).to_numpy()
    predictions[closed] = 0.0
    return pd.DataFrame({
        "date": target_dates,
        "horizon": horizons,
        "prediction": predictions,
    })


def naive_seasonal_fallback_forecast(
    daily: pd.DataFrame,
    origin: str | pd.Timestamp,
    horizon: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Prévoit chaque jour par le même jour des semaines passées, avec repli si jour fermé."""
    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif")
    origin_timestamp = pd.Timestamp(origin).normalize()
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.loc[work["date"] <= origin_timestamp].sort_values("date")
    if work.empty:
        raise ValueError("Aucune donnée disponible jusqu'à l'origine")
    revenue_by_date = work.drop_duplicates("date").set_index("date")["revenue"].astype(float)
    target_dates = pd.date_range(
        origin_timestamp + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    horizons = np.arange(1, horizon + 1)
    target_closed = is_known_closed(target_dates, config).to_numpy()

    min_date = revenue_by_date.index.min()
    predictions = np.zeros(horizon, dtype=float)

    for i, (target_date, h) in enumerate(zip(target_dates, horizons)):
        if target_closed[i]:
            predictions[i] = 0.0
            continue
        k = int(np.ceil(h / 7))
        while True:
            source_date = target_date - pd.Timedelta(days=7 * k)
            if source_date < min_date:
                raise ValueError(
                    f"La date source du repli tombe avant le début de la série: {source_date.date()}"
                )
            if source_date not in revenue_by_date.index:
                raise ValueError(
                    f"Date source manquante pour le repli: {source_date.date()}"
                )
            if not is_known_closed([source_date], config).iloc[0]:
                predictions[i] = float(revenue_by_date.loc[source_date])
                break
            k += 1

    return pd.DataFrame({
        "date": target_dates,
        "horizon": horizons,
        "prediction": predictions,
    })

