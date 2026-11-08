"""Tests de la série journalière et des anomalies."""

import pandas as pd

from retail_forecast.data.aggregate import (
    daily_revenue,
    detect_anomalous_days,
    weekly_revenue,
)
from retail_forecast.data.load import load_config


def test_daily_calendar_and_partial_day():
    config = load_config()
    transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2010-12-01 09:00", "2010-12-02 09:00", "2010-12-04 09:00"]
            ),
            "Revenue": [10.0, 20.0, 5.0],
        }
    )
    daily = daily_revenue(transactions, config)
    assert daily["date"].min() == pd.Timestamp("2010-12-01")
    assert daily["date"].max() == pd.Timestamp("2011-12-08")
    assert len(daily) == 373
    assert daily["date"].is_monotonic_increasing
    assert daily["date"].is_unique
    assert not daily["is_partial_day"].any()
    saturday_with_transactions = daily.loc[
        daily["date"].eq("2010-12-04")
    ].iloc[0]
    assert saturday_with_transactions["is_trading_day"]


def test_closure_dates_are_non_trading_but_transactions_win():
    config = load_config()
    transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2010-12-24 09:00", "2010-12-25 09:00"]
            ),
            "Revenue": [10.0, 20.0],
        }
    )
    daily = daily_revenue(transactions, config)
    closure = daily.loc[daily["date"].eq("2010-12-24")].iloc[0]
    saturday_with_transactions = daily.loc[
        daily["date"].eq("2010-12-25")
    ].iloc[0]
    assert closure["n_transactions"] == 1
    assert closure["is_trading_day"]
    assert saturday_with_transactions["n_transactions"] == 1
    assert saturday_with_transactions["is_trading_day"]

    no_transactions = daily_revenue(
        pd.DataFrame(
            {
                "InvoiceDate": pd.to_datetime(["2010-12-23"]),
                "Revenue": [1.0],
            }
        ),
        config,
    )
    assert not no_transactions.loc[
        no_transactions["date"].eq("2010-12-24"), "is_trading_day"
    ].iloc[0]


def test_daily_sum_equals_cleaned_revenue_except_partial_day():
    config = load_config()
    transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2011-12-08 09:00", "2011-12-09 09:00"]
            ),
            "Revenue": [10.0, 20.0],
        }
    )
    daily = daily_revenue(transactions, config)
    assert daily["revenue"].sum() == 10.0


def test_weekly_marks_first_and_last_partial():
    config = load_config()
    daily = daily_revenue(
        pd.DataFrame(
            {
                "InvoiceDate": pd.to_datetime(["2010-12-01", "2011-12-08"]),
                "Revenue": [1.0, 2.0],
            }
        ),
        config,
    )
    weekly = weekly_revenue(daily)
    assert weekly.iloc[0]["is_partial_week"]
    assert weekly.iloc[-1]["is_partial_week"]


def test_detect_anomalous_days():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-01", periods=70),
            "revenue": [100.0] * 69 + [10000.0],
        }
    )
    anomalies = detect_anomalous_days(daily, window=7, k=5)
    assert not anomalies.empty
    assert anomalies.iloc[-1]["date"] == pd.Timestamp("2011-03-11")
