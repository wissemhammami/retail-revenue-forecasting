"""Tests for time splitting."""

import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.splits import build_folds


def _daily() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2010-12-01", "2011-12-08")})


def test_folds_are_spaced_bounded_and_regimes_are_correct():
    config = load_config()
    dev = build_folds(_daily(), config, "dev")
    test = build_folds(_daily(), config, "test")
    assert all(fold["target_end"] <= pd.Timestamp("2011-12-08") for fold in dev + test)
    assert all(fold["target_end"] <= pd.Timestamp("2011-08-31") for fold in dev)
    assert all(fold["train_days"] >= 90 for fold in dev + test)
    assert all(
        (current["origin"] - previous["origin"]).days == 7
        for folds in (dev, test)
        for previous, current in zip(folds, folds[1:])
    )
    assert set(fold["regime"] for fold in dev) == {"dev"}
    assert set(fold["regime"] for fold in test) == {
        "normal", "montee_novembre", "decembre"
    }
    assert all(
        fold["regime"] == (
            "decembre"
            if fold["target_end"] > pd.Timestamp("2011-12-01")
            else "normal"
            if fold["origin"] <= pd.Timestamp("2011-10-15")
            else "montee_novembre"
        )
        for fold in test
    )
    assert {fold["overlap_days_previous"] for fold in dev[1:]} == {21}


def test_dev_and_test_target_windows_do_not_overlap():
    config = load_config()
    dev = build_folds(_daily(), config, "dev")
    test = build_folds(_daily(), config, "test")
    dev_targets = {date for fold in dev for date in pd.date_range(fold["target_start"], fold["target_end"])}
    test_targets = {date for fold in test for date in pd.date_range(fold["target_start"], fold["target_end"])}
    assert dev_targets.isdisjoint(test_targets)
