import pandas as pd
import pytest

from retail_forecast.evaluation.backtest import run_backtest


CONFIG = {"evaluation": {"horizon": 3}}


def _daily() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2020-01-10")
    return pd.DataFrame({"date": dates, "revenue": range(10)})


def _forecast(daily, origin, horizon, config):
    dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"date": dates, "horizon": range(1, horizon + 1), "prediction": 1.0})


def _forecast_other(daily, origin, horizon, config):
    result = _forecast(daily, origin, horizon, config)
    result["prediction"] = 2.0
    return result


def test_backtest_has_common_dates_for_forecasters() -> None:
    folds = [{
        "kind": "dev", "origin": pd.Timestamp("2020-01-03"),
        "regime": "dev",
    }]
    left = run_backtest(_forecast, _daily(), folds, CONFIG)
    right = run_backtest(_forecast_other, _daily(), folds, CONFIG)
    assert left[["kind", "origin", "date", "horizon", "regime"]].equals(
        right[["kind", "origin", "date", "horizon", "regime"]]
    )


def test_backtest_raises_when_actual_is_missing() -> None:
    folds = [{
        "kind": "dev", "origin": pd.Timestamp("2020-01-07"),
        "regime": "dev",
    }]
    with pytest.raises(ValueError, match="Actual revenue missing"):
        run_backtest(_forecast, _daily().iloc[:-1], folds, CONFIG)
