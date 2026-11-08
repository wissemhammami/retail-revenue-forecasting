"""Tests for champion selection and qualification rules."""

from __future__ import annotations

import pandas as pd
import pytest

from retail_forecast.evaluation.selection import (
    count_better_origins,
    is_qualified,
    select_champion,
)


def _origins_df(wapes: list[float]) -> pd.DataFrame:
    origins = pd.date_range("2011-03-01", periods=len(wapes), freq="7D")
    return pd.DataFrame({"origin": origins, "wape": wapes})


def test_12_out_of_23_qualifies_when_pooled_lower():
    # 12/23 qualifies when pooled WAPE is lower.
    assert is_qualified(
        challenger_pooled_wape=0.35,
        baseline_pooled_wape=0.362,
        n_better_origins=12,
        total_origins=23,
    ) is True


def test_11_out_of_23_does_not_qualify():
    # 11/23 does not qualify.
    assert is_qualified(
        challenger_pooled_wape=0.35,
        baseline_pooled_wape=0.362,
        n_better_origins=11,
        total_origins=23,
    ) is False


def test_pooled_lower_but_11_origins_does_not_qualify():
    # Lower pooled WAPE but 11 origins: no.
    assert is_qualified(
        challenger_pooled_wape=0.30,
        baseline_pooled_wape=0.362,
        n_better_origins=11,
        total_origins=23,
    ) is False


def test_12_origins_but_pooled_higher_does_not_qualify():
    # 12 origins but higher pooled WAPE: no.
    assert is_qualified(
        challenger_pooled_wape=0.38,
        baseline_pooled_wape=0.362,
        n_better_origins=12,
        total_origins=23,
    ) is False


def test_ties_do_not_count_in_better_origins():
    # Ties do not count.
    base_wapes = [0.30] * 23
    # 10 better origins (0.20), 5 ties (0.30), 8 worse (0.40).
    chall_wapes = [0.20] * 10 + [0.30] * 5 + [0.40] * 8
    b_df = _origins_df(base_wapes)
    c_df = _origins_df(chall_wapes)

    better = count_better_origins(c_df, b_df)
    assert better == 10
    # With 10 < 12, not qualified.
    assert is_qualified(0.28, 0.30, better, 23) is False


def test_two_qualified_challengers_lowest_pooled_wins():
    # Two qualified challengers: lowest pooled WAPE wins.
    candidates = [
        {"name": "baseline", "pooled_wape": 0.3620, "qualified": True},
        {"name": "prophet", "pooled_wape": 0.3400, "qualified": True},
        {"name": "xgboost", "pooled_wape": 0.3200, "qualified": True},
    ]
    res = select_champion(candidates)
    assert res["champion_name"] == "xgboost"
    assert res["champion_cand"]["name"] == "xgboost"


def test_no_challenger_qualified_baseline_wins():
    # None qualified: baseline is champion.
    candidates = [
        {"name": "baseline", "pooled_wape": 0.3620, "qualified": True},
        {"name": "prophet", "pooled_wape": 0.3500, "qualified": False},
        {"name": "xgboost", "pooled_wape": 0.3300, "qualified": False},
    ]
    res = select_champion(candidates)
    assert res["champion_name"] == "baseline"
    assert "a simple rule is enough" in res["rule_applied"]


def test_unaligned_origins_raises_error():
    # Unaligned origins: error.
    b_df = _origins_df([0.3] * 23)
    c_df = _origins_df([0.2] * 23)
    # Shift the challenger dates.
    c_df["origin"] = pd.date_range("2011-03-08", periods=23, freq="7D")

    with pytest.raises(ValueError, match="are not identical"):
        count_better_origins(c_df, b_df)
