import pandas as pd
import pytest
import numpy as np
from pathlib import Path

from retail_forecast.evaluation.closures import is_known_closed
from retail_forecast.data.load import load_config
from retail_forecast.models.baseline import (
    naive_seasonal_forecast,
    naive_seasonal_fallback_forecast,
)


CONFIG = {
    "aggregation": {"closure_dates": ["2020-01-03"]},
    "evaluation": {"horizon": 28},
}
MAP_CONFIG = {"aggregation": {"closure_dates": []}, "evaluation": {"horizon": 28}}


def _daily() -> pd.DataFrame:
    dates = pd.date_range("2019-12-01", "2020-02-01")
    return pd.DataFrame({
        "date": dates,
        "revenue": (dates.dayofyear * 10).astype(float),
        "is_trading_day": np.asarray(dates.dayofweek) != 5,
    })


def test_naive_is_exact_for_weekly_periodic_series() -> None:
    daily = _daily()
    daily["revenue"] = daily.index.to_numpy() % 7
    result = naive_seasonal_forecast(daily, "2019-12-20", 28, CONFIG)
    assert result["prediction"].notna().all()
    expected = daily.set_index("date").loc[result["date"], "revenue"].to_numpy(copy=True)
    expected[is_known_closed(result["date"], CONFIG).to_numpy()] = 0
    assert result["prediction"].to_numpy() == pytest.approx(expected)


def test_horizon_to_week_mapping_and_saturday_zero() -> None:
    daily = _daily()
    result = naive_seasonal_forecast(daily, "2019-12-20", 28, CONFIG)
    assert result.loc[result["horizon"].isin([1, 7]), "date"].sub(pd.Timestamp("2019-12-20")).dt.days.tolist() == [1, 7]
    assert result.loc[result["date"].dt.dayofweek.eq(5), "prediction"].eq(0).all()


def test_all_requested_horizon_to_k_mappings() -> None:
    daily = _daily()
    daily["revenue"] = np.arange(len(daily), dtype=float)
    origin = pd.Timestamp("2019-12-19")
    result = naive_seasonal_forecast(daily, origin, 28, MAP_CONFIG)
    by_date = daily.set_index("date")["revenue"]
    for h in [1, 7, 8, 14, 15, 21, 22, 28]:
        k = (h + 6) // 7
        target = origin + pd.Timedelta(days=h)
        source = target - pd.Timedelta(days=7 * k)
        observed = result.loc[result["horizon"].eq(h), "prediction"].iloc[0]
        assert observed == by_date.loc[source]


def test_naive_has_no_future_leakage() -> None:
    daily = _daily()
    altered = daily.copy()
    altered.loc[altered["date"] > "2019-12-20", "revenue"] = 999999
    left = naive_seasonal_forecast(daily, "2019-12-20", 28, CONFIG)
    right = naive_seasonal_forecast(altered, "2019-12-20", 28, CONFIG)
    assert left["prediction"].tolist() == right["prediction"].tolist()


def test_naive_raises_when_source_is_before_series() -> None:
    with pytest.raises(ValueError, match="falls before the start"):
        naive_seasonal_forecast(_daily(), "2019-12-01", 28, CONFIG)


def test_known_closed_matches_daily_trading_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "config.yaml")
    daily = pd.read_csv(
        root / "data" / "processed" / "daily_revenue.csv",
        parse_dates=["date"],
    )
    known_closed = is_known_closed(daily["date"], config)
    assert (known_closed == ~daily["is_trading_day"].astype(bool)).all()


def test_fallback_identical_to_strict_without_closures() -> None:
    # Periodic weekly series without closures: fallback matches strict naive.
    daily = _daily()
    daily["revenue"] = daily.index.to_numpy() % 7
    strict = naive_seasonal_forecast(daily, "2019-12-20", 28, MAP_CONFIG)
    fallback = naive_seasonal_fallback_forecast(daily, "2019-12-20", 28, MAP_CONFIG)
    pd.testing.assert_frame_equal(strict, fallback)


def test_fallback_uses_previous_week_when_source_is_closed() -> None:
    # A closure on a source date: fallback uses the previous week; strict naive predicts 0.
    # Origin 2019-12-20, h=1 -> target 2019-12-21 (closed Saturday), h=7 -> target 2019-12-27 (Friday).
    # k=1 for h=7 -> source date = 2019-12-27 - 7d = 2019-12-20 (Friday).
    # If 2019-12-20 is declared closed:
    # Strict naive uses 2019-12-20 (whose revenue is 0 because it is known closed).
    # The strict naive source is not forced to 0 when the target is open, unless the target itself is closed.
    # If the source is a public holiday, its actual revenue is 0.
    # Therefore strict naive predicts 0. Fallback uses 2019-12-13 (previous open week, revenue > 0).
    cfg = {
        "aggregation": {"closure_dates": ["2019-12-20"]},
        "evaluation": {"horizon": 28},
    }
    dates = pd.date_range("2019-12-01", "2020-01-15")
    # Set positive revenue everywhere except 2019-12-20, where revenue = 0 because it is closed.
    daily = pd.DataFrame({
        "date": dates,
        "revenue": 100.0,
        "is_trading_day": True,
    })
    daily.loc[daily["date"] == "2019-12-20", "revenue"] = 0.0
    daily.loc[daily["date"] == "2019-12-13", "revenue"] = 42.0

    strict = naive_seasonal_forecast(daily, "2019-12-20", 7, cfg)
    fallback = naive_seasonal_fallback_forecast(daily, "2019-12-20", 7, cfg)

    # For h=7 (target 2019-12-27, open Friday).
    # Strict source (k=1) = 2019-12-20 -> strict predicts 0.0.
    assert strict.loc[strict["horizon"] == 7, "prediction"].iloc[0] == 0.0
    # Fallback skips 2019-12-20 (closed) and uses k=2 -> 2019-12-13 -> predicts 42.0.
    assert fallback.loc[fallback["horizon"] == 7, "prediction"].iloc[0] == 42.0


def test_fallback_known_closed_target_predicts_zero() -> None:
    # Known closed target: both forecasts are 0.
    cfg = {
        "aggregation": {"closure_dates": ["2019-12-25"]},
        "evaluation": {"horizon": 28},
    }
    daily = _daily()
    strict = naive_seasonal_forecast(daily, "2019-12-20", 7, cfg)
    fallback = naive_seasonal_fallback_forecast(daily, "2019-12-20", 7, cfg)

    # 2019-12-25 is h=5.
    target_date = pd.Timestamp("2019-12-25")
    assert strict.loc[strict["date"] == target_date, "prediction"].iloc[0] == 0.0
    assert fallback.loc[fallback["date"] == target_date, "prediction"].iloc[0] == 0.0


def test_fallback_has_no_future_leakage() -> None:
    # Leakage prevention: changing revenue after the origin does not change fallback predictions.
    daily = _daily()
    altered = daily.copy()
    altered.loc[altered["date"] > "2019-12-20", "revenue"] = 999999.0
    left = naive_seasonal_fallback_forecast(daily, "2019-12-20", 28, CONFIG)
    right = naive_seasonal_fallback_forecast(altered, "2019-12-20", 28, CONFIG)
    pd.testing.assert_frame_equal(left, right)


def test_fallback_raises_when_source_is_before_series() -> None:
    # Explicit error when fallback goes outside the series.
    with pytest.raises(ValueError, match="falls before the start"):
        naive_seasonal_fallback_forecast(_daily(), "2019-12-01", 28, CONFIG)
