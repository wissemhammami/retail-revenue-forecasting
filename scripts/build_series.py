"""Construit les séries journalière et hebdomadaire."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.aggregate import (
    daily_revenue,
    detect_anomalous_days,
    weekly_revenue,
)
from retail_forecast.data.load import load_config


def atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    """Écrit atomiquement un CSV."""
    temporary = destination.with_name(destination.name + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    """Construit et écrit les séries demandées."""
    config = load_config()
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    dashboard = Path(__file__).resolve().parents[1] / "dashboard_data"
    processed.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(
        interim / "clean_transactions.csv",
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
        parse_dates=["InvoiceDate"],
    )
    daily = daily_revenue(cleaned, config)
    weekly = weekly_revenue(daily)
    anomalies = detect_anomalous_days(
        daily, k=config["aggregation"]["anomaly_k"]
    )
    daily_output = daily[["date", "revenue", "is_trading_day"]]
    atomic_write(daily_output, processed / "daily_revenue.csv")
    atomic_write(weekly, processed / "weekly_revenue.csv")
    atomic_write(anomalies, processed / "anomalous_days.csv")
    atomic_write(daily_output, dashboard / "daily_revenue.csv")
    print(f"daily_start={daily['date'].min().date()}")
    print(f"daily_end={daily['date'].max().date()}")
    print(f"daily_days={len(daily)}")
    print(f"weekly_rows={len(weekly)}")
    print(f"anomalous_days={len(anomalies)}")
    print(daily.groupby(daily["date"].dt.day_name())["n_transactions"].sum().to_string())


if __name__ == "__main__":
    main()
