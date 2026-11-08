"""Tests des features calendaires et anti-fuite."""

import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.features.build_features import build_features, features_for_origin


def _daily() -> pd.DataFrame:
    dates = pd.date_range("2010-12-01", periods=70)
    return pd.DataFrame({
        "date": dates,
        "revenue": range(70),
        "is_trading_day": True,
    })


def test_lags_and_rolling_are_shifted():
    features = build_features(_daily(), load_config())
    assert pd.isna(features.iloc[6]["lag_7"])
    assert features.iloc[7]["lag_7"] == 0
    assert pd.isna(features.iloc[6]["rolling_mean_7"])
    assert features.iloc[7]["rolling_mean_7"] == 3


def test_future_revenue_cannot_change_historical_features():
    daily = _daily()
    baseline = build_features(daily, load_config())
    altered = daily.copy()
    altered.loc[altered["date"] >= pd.Timestamp("2011-01-15"), "revenue"] = 999999
    changed = build_features(altered, load_config())
    assert baseline.loc[: "2011-01-14"].equals(changed.loc[: "2011-01-14"])


def test_features_csv_lag_7_uses_only_revenue_on_or_before_date():
    daily = _daily()
    baseline = build_features(daily, load_config())
    for date in baseline.index[7:]:
        altered = daily.copy()
        altered.loc[altered["date"] > date, "revenue"] = 999999
        changed = build_features(altered, load_config())
        assert baseline.loc[date, "lag_7"] == changed.loc[date, "lag_7"]


def test_features_for_origin_ignores_revenue_after_origin():
    daily = _daily()
    origin = pd.Timestamp("2011-01-15")
    baseline = features_for_origin(daily, origin, 28)
    altered = daily.copy()
    altered.loc[altered["date"] > origin, "revenue"] = 987654
    changed = features_for_origin(altered, origin, 28)
    pd.testing.assert_frame_equal(baseline, changed)


def test_christmas_calendar_flags():
    dates = pd.date_range("2011-12-16", "2012-01-18")
    daily = pd.DataFrame({"date": dates, "revenue": 1.0, "is_trading_day": True})
    features = build_features(daily, load_config())
    assert features.loc["2011-12-17":"2011-12-23", "is_pre_christmas"].all()
    assert not features.loc["2011-12-24", "is_pre_christmas"]
    assert features.loc["2012-01-04":"2012-01-18", "is_post_christmas"].all()
