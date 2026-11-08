"""Découpage temporel sans fuite entre développement et test."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd


def generate_origins(start: str | date | pd.Timestamp,
                     end: str | date | pd.Timestamp,
                     step: int) -> list[pd.Timestamp]:
    """Retourne des origines inclusives régulièrement espacées."""
    if step <= 0:
        raise ValueError("step doit être strictement positif")
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{step}D"))


def _evaluation_dates(config: Mapping[str, Any], kind: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Lit la fenêtre d'origines du développement ou du test."""
    if kind not in {"dev", "test"}:
        raise ValueError("kind doit valoir 'dev' ou 'test'")
    section = config["evaluation"][f"{kind}_origins"]
    return pd.Timestamp(section["start"]), pd.Timestamp(section["end"])


def _overlap_days(previous: pd.Timestamp, current: pd.Timestamp, horizon: int) -> int:
    """Calcule les dates cibles partagées par deux origines consécutives."""
    previous_targets = pd.date_range(previous + pd.Timedelta(days=1),
                                     previous + pd.Timedelta(days=horizon))
    current_targets = pd.date_range(current + pd.Timedelta(days=1),
                                    current + pd.Timedelta(days=horizon))
    return len(previous_targets.intersection(current_targets))


def build_folds(daily: pd.DataFrame, config: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    """Construit des folds expansifs et vérifie les bornes temporelles."""
    evaluation = config["evaluation"]
    horizon = int(evaluation["horizon"])
    last_usable = pd.Timestamp(evaluation["last_usable_date"])
    min_train_days = int(evaluation["min_train_days"])
    start, end = _evaluation_dates(config, kind)
    origins = generate_origins(start, end, int(evaluation["origin_step_days"]))
    dates = pd.to_datetime(daily["date"]).dt.normalize()
    first_date = dates.min()
    folds: list[dict[str, Any]] = []
    for origin in origins:
        target_start = origin + pd.Timedelta(days=1)
        target_end = origin + pd.Timedelta(days=horizon)
        if kind == "dev" and target_end > pd.Timestamp("2011-08-31"):
            continue
        if target_end > last_usable:
            raise ValueError(
                f"Le fold {origin.date()} dépasse last_usable_date "
                f"({target_end.date()} > {last_usable.date()})"
            )
        train_days = int((dates <= origin).sum())
        if train_days < min_train_days:
            raise ValueError(
                f"Historique insuffisant pour {origin.date()}: "
                f"{train_days} jours, minimum {min_train_days}"
            )
        if origin < first_date:
            raise ValueError(f"Origine hors historique: {origin.date()}")
        if kind == "dev":
            regime = "dev"
        elif target_end > pd.Timestamp(evaluation["december_regime_target_date"]):
            regime = "decembre"
        elif origin <= pd.Timestamp(evaluation["test_regime_split"]):
            regime = "normal"
        else:
            regime = "montee_novembre"
        overlap = (
            _overlap_days(folds[-1]["origin"], origin, horizon)
            if folds else 0
        )
        folds.append({
            "kind": kind,
            "origin": origin,
            "train_end": origin,
            "target_start": target_start,
            "target_end": target_end,
            "regime": regime,
            "train_days": train_days,
            "overlap_days_previous": overlap,
        })
    return folds


def overlap_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Résume le chevauchement des fenêtres d'un ensemble de folds."""
    overlaps = [int(fold["overlap_days_previous"]) for fold in folds[1:]]
    return {
        "folds": len(folds),
        "overlap_days_consecutive": overlaps,
        "overlap_days_unique": sorted(set(overlaps)),
    }
