import pandas as pd
import pytest

from retail_forecast.evaluation.metrics import (
    bias,
    horizon_bucket,
    mae,
    per_origin_wape,
    rmse,
    summarize,
    wape,
)


def test_metrics_known_values_and_bias_sign() -> None:
    actual = [10, 20]
    prediction = [12, 16]
    assert wape(actual, prediction) == pytest.approx(6 / 30)
    assert mae(actual, prediction) == pytest.approx(3)
    assert rmse(actual, prediction) == pytest.approx((10**0.5))
    assert bias(actual, prediction) == pytest.approx(-1)
    with pytest.raises(ValueError, match="sum of actual values"):
        wape([0, 0], [1, 2])


def test_metrics_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mae([], [])
    with pytest.raises(ValueError, match="must not be empty"):
        rmse([], [])
    with pytest.raises(ValueError, match="must not be empty"):
        bias([], [])


def test_horizon_buckets_and_pooled_summary() -> None:
    buckets = [[1, 7], [8, 14], [15, 28]]
    assert [horizon_bucket(h, buckets) for h in [1, 7, 8, 14, 15, 21, 22, 28]] == [
        "1-7", "1-7", "8-14", "8-14", "15-28", "15-28", "15-28", "15-28"
    ]
    frame = pd.DataFrame({
        "origin": ["a", "a", "b"],
        "actual": [10, 20, 10],
        "prediction": [12, 16, 10],
    })
    pooled = summarize(frame)
    assert pooled.loc[0, "wape"] == pytest.approx(6 / 40)
    assert len(per_origin_wape(frame)) == 2
