"""Métriques de prévision et agrégations de backtests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _arrays(actual: Iterable[float], pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convertit deux vecteurs, vérifie leur longueur et leur non-vacuité."""
    actual_array = np.asarray(list(actual), dtype=float)
    pred_array = np.asarray(list(pred), dtype=float)
    if actual_array.shape != pred_array.shape:
        raise ValueError("actual et pred doivent avoir la même longueur")
    if actual_array.size == 0:
        raise ValueError("actual et pred ne doivent pas être vides")
    return actual_array, pred_array


def wape(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Retourne l'erreur absolue pondérée par le revenu réel.

    Si la somme absolue des valeurs réelles vaut zéro, le WAPE est
    mathématiquement indéfini et lève ValueError plutôt que de produire
    une valeur artificielle. Les métriques MAE, RMSE et biais restent
    définies lorsque les valeurs réelles sont toutes nulles.
    """
    actual_array, pred_array = _arrays(actual, pred)
    denominator = float(np.abs(actual_array).sum())
    if denominator == 0:
        raise ValueError("WAPE indéfini : la somme des valeurs réelles vaut zéro")
    return float(np.abs(pred_array - actual_array).sum() / denominator)


def mae(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Retourne l'erreur absolue moyenne."""
    actual_array, pred_array = _arrays(actual, pred)
    return float(np.abs(pred_array - actual_array).mean())


def rmse(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Retourne la racine de l'erreur quadratique moyenne."""
    actual_array, pred_array = _arrays(actual, pred)
    return float(np.sqrt(np.square(pred_array - actual_array).mean()))


def bias(actual: Iterable[float], pred: Iterable[float]) -> float:
    """Retourne le biais moyen, positif en cas de sur-prévision."""
    actual_array, pred_array = _arrays(actual, pred)
    return float((pred_array - actual_array).mean())


def horizon_bucket(h: int, buckets: Iterable[Iterable[int]]) -> str:
    """Associe un horizon à un libellé de tranche configurée."""
    horizon = int(h)
    for bucket in buckets:
        start, end = (int(value) for value in bucket)
        if start <= horizon <= end:
            return f"{start}-{end}"
    raise ValueError(f"Horizon hors des tranches configurées: {horizon}")


def summarize(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Agrège les métriques sur tous les couples origine-date de chaque groupe."""
    group_columns = [by] if isinstance(by, str) else (by or [])
    required = {"actual", "prediction", *group_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour summarize: {sorted(missing)}")

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
    """Retourne le WAPE de chaque origine."""
    if "origin" not in df.columns:
        raise ValueError("La colonne origin est requise pour per_origin_wape")
    return (
        df.groupby("origin", sort=True)
        .apply(lambda group: wape(group["actual"], group["prediction"]), include_groups=False)
        .rename("wape")
        .reset_index()
    )
