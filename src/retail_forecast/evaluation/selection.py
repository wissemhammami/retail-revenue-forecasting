"""Champion selection and decision rules."""

from __future__ import annotations

from typing import Any
import pandas as pd


def count_better_origins(
    challenger_per_origin: pd.DataFrame,
    baseline_per_origin: pd.DataFrame,
) -> int:
    """Count origins where the challenger has strictly lower WAPE than the baseline.

    Verify that origins are exactly identical between both series.
    """
    for df, name in [(challenger_per_origin, "challenger_per_origin"), (baseline_per_origin, "baseline_per_origin")]:
        if "origin" not in df.columns or "wape" not in df.columns:
            raise ValueError(f"{name} must contain the 'origin' and 'wape' columns")

    c_sorted = challenger_per_origin.sort_values("origin").reset_index(drop=True)
    b_sorted = baseline_per_origin.sort_values("origin").reset_index(drop=True)

    c_origins = pd.to_datetime(c_sorted["origin"]).dt.normalize()
    b_origins = pd.to_datetime(b_sorted["origin"]).dt.normalize()

    if not c_origins.equals(b_origins):
        raise ValueError("Challenger and baseline origins are not identical")

    # Strictly lower (<); ties do not count.
    strictly_better = (c_sorted["wape"].astype(float) < b_sorted["wape"].astype(float)).sum()
    return int(strictly_better)


def is_qualified(
    challenger_pooled_wape: float,
    baseline_pooled_wape: float,
    n_better_origins: int,
    total_origins: int = 23,
) -> bool:
    """Determine whether a challenger qualifies against the baseline.

    Conditions:
    - Pooled WAPE strictly lower than the baseline.
    - WAPE strictly lower by origin for a majority of origins (>= 12 of 23).
    """
    majority_threshold = (total_origins // 2) + 1
    has_lower_pooled = float(challenger_pooled_wape) < float(baseline_pooled_wape)
    has_majority = int(n_better_origins) >= majority_threshold
    return bool(has_lower_pooled and has_majority)


def select_champion(
    candidates: list[dict[str, Any]],
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    """Select the champion among the baseline and qualified challengers.

    Each candidate is a dictionary containing at least:
    - name: str (for example, 'baseline', 'prophet', 'xgboost')
    - pooled_wape: float
    - qualified: bool (for challengers; the baseline is qualified by default)

    Rule:
    - Among qualified challengers, the lowest pooled WAPE wins.
    - If no challenger qualifies, the baseline is champion ("a simple rule is enough").
    """
    baseline_cand = next((c for c in candidates if c["name"] == baseline_name), None)
    if baseline_cand is None:
        raise ValueError(f"Baseline candidate '{baseline_name}' not found")

    qualified_challengers = [
        c for c in candidates if c["name"] != baseline_name and c.get("qualified", False)
    ]

    if not qualified_challengers:
        champion = baseline_cand
        rule_applied = "No challenger qualifies: the baseline is champion ('a simple rule is enough')."
    else:
        # Among qualified challengers, the lowest WAPE wins.
        champion = min(qualified_challengers, key=lambda c: float(c["pooled_wape"]))
        rule_applied = (
            f"Qualified challenger with the lowest pooled WAPE ({champion['name']} at {champion['pooled_wape']:.4f})."
        )

    return {
        "champion_name": champion["name"],
        "champion_cand": champion,
        "rule_applied": rule_applied,
    }
