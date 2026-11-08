"""Tests unitaires du nettoyage, sans réseau."""

import pandas as pd
import pytest

from retail_forecast.data.clean import (
    add_revenue,
    clean_pipeline,
    drop_cancelled_orders,
    drop_exact_duplicates,
    flag_cancellations,
    remove_invalid_prices,
    remove_invalid_quantities,
    remove_non_product_codes,
    remove_non_product_invoices,
    winsorize,
)
from retail_forecast.data.load import load_config


@pytest.fixture
def config():
    return load_config()


def frame():
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "C2", "3", "4", "5", "6"],
            "StockCode": ["A", "A", "A", "POST", "gift_0001x", "b", "C"],
            "CustomerID": ["10", "10", "10", "11", "12", None, "13"],
            "Quantity": [2, 3, -2, 1, 1, 0, -1],
            "UnitPrice": [2.0, 3.0, 2.0, 1.0, 1.0, 0.0, 1.0],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01 09:00",
                    "2011-01-01 10:00",
                    "2011-01-01 11:00",
                    "2011-01-01 12:00",
                    "2011-01-01 13:00",
                    "2011-01-01 14:00",
                    "2011-01-01 15:00",
                ]
            ),
        }
    )


def test_each_cleaning_function(config):
    source = frame()
    assert len(drop_exact_duplicates(pd.concat([source, source.iloc[[0]]]))) == 7
    assert flag_cancellations(source)["is_cancellation"].sum() == 1
    assert len(remove_invalid_prices(source)) == 6
    assert len(remove_non_product_codes(source, ["post"], ["GIFT_0001"])) == 5
    assert len(
        remove_non_product_invoices(
            pd.DataFrame({"InvoiceNo": ["A1", "a2", "1"]}), ["A"]
        )
    ) == 1
    assert len(remove_invalid_quantities(source, "exclude")) == 4
    assert len(remove_invalid_quantities(source, "net")) == 6
    capped = winsorize(source.loc[[0, 1]], config["cleaning"])
    assert {"quantity_capped", "price_capped"} <= set(capped)
    assert add_revenue(capped).loc[0, "Revenue"] == pytest.approx(4.0)


def test_drop_cancelled_orders_matches_one_latest_order():
    source = frame()
    result = drop_cancelled_orders(source)
    assert list(result["InvoiceNo"]) == ["1", "C2", "3", "4", "5", "6"]


def test_drop_cancelled_orders_does_not_match_without_customer():
    source = frame().iloc[[5, 6]].copy()
    result = drop_cancelled_orders(source)
    assert len(result) == 2


def test_clean_pipeline_exclude_and_summary(config):
    cleaned, summary = clean_pipeline(frame(), config)
    assert not cleaned["InvoiceNo"].astype("string").str.startswith("C").any()
    assert len(cleaned) == 1
    assert summary[0]["rows_before"] == len(frame())
    assert all(
        summary[index]["rows_after"] == summary[index + 1]["rows_before"]
        for index in range(len(summary) - 1)
    )
    assert len(frame()) - sum(step["removed"] for step in summary) == len(cleaned)
    assert (cleaned["Revenue"] == cleaned["Quantity"] * cleaned["UnitPrice"]).all()


def test_clean_pipeline_net_keeps_returns(config):
    net_config = {**config, "cleaning": {**config["cleaning"], "returns_mode": "net"}}
    cleaned, _ = clean_pipeline(frame(), net_config)
    assert cleaned["InvoiceNo"].astype("string").str.startswith("C").any()
    assert (cleaned["Revenue"] == cleaned["Quantity"] * cleaned["UnitPrice"]).all()


def test_clean_pipeline_disabled_winsorization_preserves_large_quantity(config):
    disabled = {
        **config,
        "cleaning": {
            **config["cleaning"],
            "winsorize": {**config["cleaning"]["winsorize"], "enabled": False},
        },
    }
    source = pd.DataFrame(
        {
            "InvoiceNo": ["large"],
            "StockCode": ["12345"],
            "CustomerID": ["10"],
            "Quantity": [500],
            "UnitPrice": [2.0],
            "InvoiceDate": pd.to_datetime(["2011-01-01"]),
        }
    )
    cleaned, summary = clean_pipeline(source, disabled)
    assert cleaned["Quantity"].max() > 100
    assert cleaned["quantity_capped"].eq(False).all()
    assert cleaned["price_capped"].eq(False).all()
    assert "drop_exact_duplicates_after_winsorization" not in {
        step["step"] for step in summary
    }
