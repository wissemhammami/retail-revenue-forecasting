from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.dashboard import _validate_forecast_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def valid_frames():
    history = pd.DataFrame({
        "date": pd.date_range("2011-12-01", periods=8),
        "revenue": [100.0] * 8,
    })
    forecast = pd.DataFrame({
        "date": pd.date_range("2011-12-09", periods=28),
        "forecast": [10.0] * 28,
        "lower_80": [5.0] * 28,
        "upper_80": [15.0] * 28,
        "reliable": [True] * 28,
    })
    closed = forecast["date"].dt.dayofweek.eq(5) | forecast["date"].between(
        "2011-12-24", "2012-01-03"
    )
    forecast.loc[closed, ["forecast", "lower_80", "upper_80"]] = 0.0
    return forecast, history


def test_rejects_missing_date(config, valid_frames):
    forecast, history = valid_frames
    forecast.loc[3, "date"] = pd.NaT
    with pytest.raises(ValueError, match="dates invalides"):
        _validate_forecast_data(forecast, history, config)


def test_rejects_duplicate_date(config, valid_frames):
    forecast, history = valid_frames
    forecast.loc[3, "date"] = forecast.loc[2, "date"]
    with pytest.raises(ValueError, match="dupliquées"):
        _validate_forecast_data(forecast, history, config)


def test_rejects_wrong_horizon(config, valid_frames):
    forecast, history = valid_frames
    with pytest.raises(ValueError, match="Horizon inattendu"):
        _validate_forecast_data(forecast.iloc[:-1], history, config)


def test_rejects_invalid_interval(config, valid_frames):
    forecast, history = valid_frames
    forecast.loc[0, "lower_80"] = 11.0
    with pytest.raises(ValueError, match="bornes"):
        _validate_forecast_data(forecast, history, config)


@pytest.mark.parametrize("date", [
    "2011-12-24", "2011-12-25", "2011-12-31", "2012-01-01", "2012-01-03",
])
def test_rejects_nonzero_configured_closure(config, valid_frames, date):
    forecast, history = valid_frames
    position = (forecast["date"] == pd.Timestamp(date)).idxmax()
    forecast.loc[position, ["forecast", "lower_80", "upper_80"]] = [1.0, 0.5, 1.5]
    with pytest.raises(ValueError, match="fermés"):
        _validate_forecast_data(forecast, history, config)


def test_allows_first_date_after_closure(config, valid_frames):
    forecast, history = valid_frames
    _validate_forecast_data(forecast, history, config)
