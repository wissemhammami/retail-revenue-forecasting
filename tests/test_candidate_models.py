"""Tests pour les modèles Prophet et XGBoost."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.backtest import run_backtest
from retail_forecast.evaluation.closures import is_known_closed
from retail_forecast.evaluation.splits import build_folds
from retail_forecast.models.baseline import (
    naive_seasonal_fallback_forecast,
    naive_seasonal_forecast,
)
from retail_forecast.models.prophet_model import forecast as prophet_forecast
from retail_forecast.models.xgb_model import forecast as xgb_forecast


@pytest.fixture(scope="module")
def data_and_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "config.yaml")
    daily = pd.read_csv(
        root / "data" / "processed" / "daily_revenue.csv",
        parse_dates=["date"],
    )
    return daily, config


@pytest.mark.parametrize("model_fn", [prophet_forecast, xgb_forecast])
def test_model_forecast_contract(data_and_config, model_fn):
    daily, config = data_and_config
    origin = "2011-06-07"
    horizon = 28
    result = model_fn(daily, origin, horizon, config)
    baseline_result = naive_seasonal_fallback_forecast(daily, origin, horizon, config)

    # Colonnes et taille
    assert list(result.columns) == ["date", "horizon", "prediction"]
    assert len(result) == 28
    # Mêmes dates que la baseline
    assert result["date"].tolist() == baseline_result["date"].tolist()
    # Aucun NaN
    assert not result["prediction"].isna().any()
    # Aucune valeur négative
    assert (result["prediction"] >= 0.0).all()
    # Jours fermés connus valent exactement 0
    closed = is_known_closed(result["date"], config)
    assert (result.loc[closed, "prediction"] == 0.0).all()


def test_end_to_end_anti_leakage(data_and_config):
    daily, config = data_and_config
    origin = "2011-06-07"
    horizon = 28

    # Création du DataFrame altéré après l'origine
    altered_daily = daily.copy()
    future_mask = altered_daily["date"] > pd.Timestamp(origin)
    altered_daily.loc[future_mask, "revenue"] = (
        altered_daily.loc[future_mask, "revenue"] * 10.0 + 5000.0
    )
    altered_daily.loc[future_mask, "is_trading_day"] = ~altered_daily.loc[
        future_mask, "is_trading_day"
    ].astype(bool)

    # Test baseline stricte et repli
    b_orig = naive_seasonal_forecast(daily, origin, horizon, config)
    b_alt = naive_seasonal_forecast(altered_daily, origin, horizon, config)
    pd.testing.assert_frame_equal(b_orig, b_alt)

    bf_orig = naive_seasonal_fallback_forecast(daily, origin, horizon, config)
    bf_alt = naive_seasonal_fallback_forecast(altered_daily, origin, horizon, config)
    pd.testing.assert_frame_equal(bf_orig, bf_alt)

    # Test Prophet
    p_orig = prophet_forecast(daily, origin, horizon, config)
    p_alt = prophet_forecast(altered_daily, origin, horizon, config)
    pd.testing.assert_frame_equal(p_orig, p_alt)

    # Test XGBoost
    x_orig = xgb_forecast(daily, origin, horizon, config)
    x_alt = xgb_forecast(altered_daily, origin, horizon, config)
    pd.testing.assert_frame_equal(x_orig, x_alt)


def test_determinism_prophet_and_xgboost(data_and_config):
    daily, config = data_and_config
    origin = "2011-06-07"
    horizon = 28

    p1 = prophet_forecast(daily, origin, horizon, config)
    p2 = prophet_forecast(daily, origin, horizon, config)
    pd.testing.assert_frame_equal(p1, p2)

    x1 = xgb_forecast(daily, origin, horizon, config)
    x2 = xgb_forecast(daily, origin, horizon, config)
    pd.testing.assert_frame_equal(x1, x2)


def test_backtest_common_dates_across_three_models(data_and_config):
    daily, config = data_and_config
    dev_folds = build_folds(daily, config, "dev")
    # Premier, médian, dernier
    first_fold = dev_folds[0]
    mid_fold = dev_folds[len(dev_folds) // 2]
    last_fold = dev_folds[-1]
    selected_folds = [first_fold, mid_fold, last_fold]

    res_base = run_backtest(naive_seasonal_fallback_forecast, daily, selected_folds, config)
    res_prophet = run_backtest(prophet_forecast, daily, selected_folds, config)
    res_xgb = run_backtest(xgb_forecast, daily, selected_folds, config)

    assert len(res_base) == 3 * 28
    assert len(res_prophet) == 3 * 28
    assert len(res_xgb) == 3 * 28

    # Mêmes couples (origin, date) pour les trois
    pairs_base = list(zip(res_base["origin"], res_base["date"]))
    pairs_prophet = list(zip(res_prophet["origin"], res_prophet["date"]))
    pairs_xgb = list(zip(res_xgb["origin"], res_xgb["date"]))

    assert pairs_base == pairs_prophet
    assert pairs_base == pairs_xgb
