"""Sélection du champion et règles de décision."""

from __future__ import annotations

from typing import Any
import pandas as pd


def count_better_origins(
    challenger_per_origin: pd.DataFrame,
    baseline_per_origin: pd.DataFrame,
) -> int:
    """Compte le nombre d'origines où le challenger a un WAPE strictement inférieur à la baseline.

    Vérifie que les origines sont rigoureusement identiques entre les deux séries.
    """
    for df, name in [(challenger_per_origin, "challenger_per_origin"), (baseline_per_origin, "baseline_per_origin")]:
        if "origin" not in df.columns or "wape" not in df.columns:
            raise ValueError(f"{name} doit contenir les colonnes 'origin' et 'wape'")

    c_sorted = challenger_per_origin.sort_values("origin").reset_index(drop=True)
    b_sorted = baseline_per_origin.sort_values("origin").reset_index(drop=True)

    c_origins = pd.to_datetime(c_sorted["origin"]).dt.normalize()
    b_origins = pd.to_datetime(b_sorted["origin"]).dt.normalize()

    if not c_origins.equals(b_origins):
        raise ValueError("Les origines du challenger et de la baseline ne sont pas identiques")

    # Strictement inférieur (<), les égalités ne comptent pas
    strictly_better = (c_sorted["wape"].astype(float) < b_sorted["wape"].astype(float)).sum()
    return int(strictly_better)


def is_qualified(
    challenger_pooled_wape: float,
    baseline_pooled_wape: float,
    n_better_origins: int,
    total_origins: int = 23,
) -> bool:
    """Détermine si un challenger est qualifié face à la baseline.

    Conditions :
    - WAPE moyen (poolé) strictement inférieur à la baseline
    - WAPE par origine strictement inférieur sur la majorité des origines (>= 12 sur 23)
    """
    majority_threshold = (total_origins // 2) + 1
    has_lower_pooled = float(challenger_pooled_wape) < float(baseline_pooled_wape)
    has_majority = int(n_better_origins) >= majority_threshold
    return bool(has_lower_pooled and has_majority)


def select_champion(
    candidates: list[dict[str, Any]],
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    """Sélectionne le champion parmi la baseline et les challengers qualifiés.

    Chaque candidat est un dictionnaire contenant au minimum :
    - name : str (ex. 'baseline', 'prophet', 'xgboost')
    - pooled_wape : float
    - qualified : bool (pour les challengers ; la baseline est réputée qualifiée par défaut)

    Règle :
    - Entre les challengers qualifiés, le WAPE poolé le plus bas l'emporte.
    - Si aucun challenger n'est qualifié, la baseline est championne (« une règle simple suffit »).
    """
    baseline_cand = next((c for c in candidates if c["name"] == baseline_name), None)
    if baseline_cand is None:
        raise ValueError(f"Le candidat baseline '{baseline_name}' est introuvable")

    qualified_challengers = [
        c for c in candidates if c["name"] != baseline_name and c.get("qualified", False)
    ]

    if not qualified_challengers:
        champion = baseline_cand
        rule_applied = "Aucun challenger n'est qualifié : la baseline est championne ('une règle simple suffit')."
    else:
        # Entre challengers qualifiés, le WAPE le plus bas gagne
        champion = min(qualified_challengers, key=lambda c: float(c["pooled_wape"]))
        rule_applied = (
            f"Challenger qualifié avec le WAPE poolé le plus bas ({champion['name']} avec {champion['pooled_wape']:.4f})."
        )

    return {
        "champion_name": champion["name"],
        "champion_cand": champion,
        "rule_applied": rule_applied,
    }
