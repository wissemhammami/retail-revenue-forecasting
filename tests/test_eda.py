"""Tests for exploratory analysis functions."""

import pandas as pd
import pytest

from retail_forecast.data.eda import (
    anomaly_details,
    cleaning_impact,
    cleaning_impact_steps,
    decompose_weekly,
    eda_summary,
    weekday_profile,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3", "4"],
            "StockCode": ["A", "B", "A", "C"],
            "Description": ["Alpha", "Beta", "Alpha", "Gamma"],
            "CustomerID": ["10", "11", "10", "12"],
            "Quantity": [1, 2, 3, 4],
            "UnitPrice": [10.0, 5.0, 10.0, 2.0],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-03", "2011-01-04", "2011-01-05", "2011-12-09"]
            ),
            "Country": ["United Kingdom", "France", "United Kingdom", "Spain"],
            "Revenue": [10.0, 10.0, 30.0, 8.0],
        }
    )


def test_eda_summary_schema_and_partial_exclusion():
    transactions = _transactions()
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-01", periods=10),
            "revenue": 1.0,
            "is_trading_day": True,
        }
    )
    summary = eda_summary(transactions, daily)
    assert list(summary.columns) == [
        "dimension",
        "key",
        "revenue",
        "n_transactions",
    ]
    assert set(summary["dimension"]) == {"weekday", "month", "country", "product"}
    assert not summary.loc[summary["dimension"].eq("weekday"), "key"].str.contains(
        "|", regex=False
    ).any()
    assert not summary["key"].astype(str).str.contains("2011-12-09").any()


def test_weekday_profile_synthetic():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2011-01-03", "2011-01-04", "2011-01-10", "2011-01-08"]
            ),
            "revenue": [10.0, 20.0, 30.0, 100.0],
            "is_trading_day": [True, True, True, False],
        }
    )
    profile = weekday_profile(daily)
    assert len(profile) == 2
    assert profile.loc[profile["weekday"].eq("Monday"), "mean"].iloc[0] == 20.0
    assert profile.loc[profile["weekday"].eq("Monday"), "median"].iloc[0] == 20.0


def test_decompose_weekly_components():
    dates = pd.date_range("2011-01-01", periods=42)
    daily = pd.DataFrame(
        {
            "date": dates,
            "revenue": [100 + i + (i % 7) for i in range(len(dates))],
            "is_trading_day": dates.dayofweek != 5,
        }
    )
    result = decompose_weekly(daily)
    assert len(result.observed) == 34
    assert len(result.trend) == 34
    assert len(result.seasonal) == 34
    assert len(result.resid) == 34


def test_anomaly_details_top_three_and_share():
    transactions = _transactions().iloc[:3].copy()
    transactions.loc[1, "InvoiceDate"] = pd.Timestamp("2011-01-03")
    anomalies = pd.DataFrame({"date": [pd.Timestamp("2011-01-03")]})
    details = anomaly_details(transactions, anomalies)
    assert len(details) == 2
    assert details.iloc[0]["Revenue"] == 10.0
    assert details["share_of_day"].sum() == pytest.approx(1.0)


def test_cleaning_impact():
    raw = _transactions()
    clean = raw.iloc[:3].copy()
    impact = cleaning_impact(raw, clean)
    assert list(impact["dataset"]) == [
        "raw",
        "raw_without_cancellations",
        "clean",
    ]
    assert list(impact["rows"]) == [4, 4, 3]
    assert list(impact["revenue"]) == [58.0, 58.0, 50.0]


def test_cleaning_impact_steps():
    config = {
        "cleaning": {
            "returns_mode": "exclude",
            "drop_cancelled_orders": False,
            "non_product_codes": [],
            "non_product_prefixes": [],
            "non_product_invoice_prefixes": [],
            "winsorize": {
                "enabled": False,
                "quantity_upper_quantile": 0.99,
                "unit_price_upper_quantile": 0.99,
            },
        }
    }
    raw = _transactions()
    summary = pd.DataFrame({"step": ["drop_exact_duplicates", "flag_cancellations"]})
    details = cleaning_impact_steps(raw, summary, config)
    assert list(details["step"]) == ["drop_exact_duplicates", "flag_cancellations"]
    assert list(details["rows_removed"]) == [0, 0]
