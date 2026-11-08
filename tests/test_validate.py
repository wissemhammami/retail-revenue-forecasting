"""Tests unitaires de validation sans accès réseau."""

import pandas as pd
import pytest

from retail_forecast.data.load import load_config
from retail_forecast.data.validate import validate_raw


@pytest.fixture
def expected():
    return load_config()["expected"] | {"n_rows": 2}


def valid_frame():
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "2"],
            "StockCode": ["A", "B"],
            "Description": ["x", "y"],
            "Quantity": [1, 2],
            "InvoiceDate": pd.to_datetime(["2010-12-01", "2011-12-08"]),
            "UnitPrice": [1.0, 2.0],
            "CustomerID": ["10", "20"],
            "Country": ["UK", "UK"],
        }
    )


def test_valid_frame_passes(expected):
    assert validate_raw(valid_frame(), expected)["rows"] == 2


@pytest.mark.parametrize("change", ["columns", "rows", "date"])
def test_invalid_frame_raises(expected, change):
    frame = valid_frame()
    if change == "columns":
        frame = frame.drop(columns=["Country"])
    elif change == "rows":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[0, "InvoiceDate"] = pd.Timestamp("2010-11-30")
    with pytest.raises(ValueError):
        validate_raw(frame, expected)


def test_load_config_resolves_paths():
    config = load_config()
    assert all(pd.io.common.is_url(value) is False for value in config["paths"].values())
    assert all(str(value).replace("\\", "/").endswith(("data/raw", "data/interim", "data/processed")) for value in config["paths"].values())
