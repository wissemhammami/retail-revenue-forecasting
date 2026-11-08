import pandas as pd
import pytest

from retail_forecast.evaluation.closures import is_reliable_after_closure
from scripts.export_forecast import _calibrate_intervals, _validate_forecast


CONFIG = {
    "aggregation": {
        "closure_dates": [
            "2020-12-24",
            "2020-12-25",
            "2020-12-26",
            "2020-12-27",
            "2020-12-28",
            "2020-12-29",
            "2020-12-30",
            "2020-12-31",
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
        ]
    }
}
BUCKETS = [[1, 2], [3, 4]]


def test_calibration_uses_open_development_residuals_by_bucket():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-06", periods=8),
            "horizon": [1, 2, 3, 4, 1, 2, 3, 4],
            "actual": [11, 22, 33, 44, 10, 20, 30, 40],
            "prediction": [10, 20, 30, 40, 10, 20, 30, 40],
        }
    )
    quantiles, counts = _calibrate_intervals(frame, CONFIG, BUCKETS, minimum_count=2)
    assert counts == {"1-2": 4, "3-4": 4}
    assert quantiles.loc["1-2", "lower_residual"] <= quantiles.loc["1-2", "upper_residual"]


def test_calibration_rejects_insufficient_bucket():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "horizon": [1, 2],
            "actual": [1, 2],
            "prediction": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="insuffisants"):
        _calibrate_intervals(frame, CONFIG, BUCKETS, minimum_count=2)


def test_forecast_validation_checks_contract_dates_and_ordering():
    valid = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "forecast": [10.0, 20.0],
            "lower_80": [5.0, 15.0],
            "upper_80": [15.0, 25.0],
            "reliable": [True, True],
        }
    )
    _validate_forecast(valid, 2)

    invalid = valid.copy()
    invalid.loc[1, "lower_80"] = 21.0
    with pytest.raises(ValueError, match="Intervalles"):
        _validate_forecast(invalid, 2)

    missing = valid.copy()
    missing.loc[0, "forecast"] = None
    with pytest.raises(ValueError, match="manquantes"):
        _validate_forecast(missing, 2)


def test_reliability_marks_after_reopening_only():
    dates = pd.date_range("2020-12-23", periods=14)
    reliable = is_reliable_after_closure(dates, CONFIG)
    assert reliable.iloc[:12].all()
    assert reliable.iloc[12:].tolist() == [False, False]


def test_reliability_without_closure_is_all_true():
    dates = pd.date_range("2021-02-01", periods=5)
    assert is_reliable_after_closure(
        dates, {"aggregation": {"closure_dates": []}}
    ).all()


def test_reliability_does_not_change_forecast_values():
    values = pd.DataFrame(
        {
            "date": pd.date_range("2020-12-23", periods=14),
            "forecast": range(14),
            "lower_80": range(14),
            "upper_80": range(1, 15),
        }
    )
    flagged = values.assign(reliable=is_reliable_after_closure(values["date"], CONFIG))
    pd.testing.assert_frame_equal(
        values,
        flagged[["date", "forecast", "lower_80", "upper_80"]],
    )
