"""Construit le résumé EDA exportable."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.eda import eda_summary
from retail_forecast.data.load import load_config, read_raw


def atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    """Écrit un CSV via un fichier temporaire."""
    temporary = destination.with_name(destination.name + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    """Génère le résumé EDA dans processed et dashboard_data."""
    config = load_config()
    interim = Path(config["paths"]["interim"])
    processed = Path(config["paths"]["processed"])
    dashboard = Path(__file__).resolve().parents[1] / "dashboard_data"
    processed.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(
        interim / "clean_transactions.csv",
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
        parse_dates=["InvoiceDate"],
    )
    daily = pd.read_csv(processed / "daily_revenue.csv", parse_dates=["date"])
    summary = eda_summary(clean, daily)
    atomic_write(summary, processed / "eda_summary.csv")
    atomic_write(summary, dashboard / "eda_summary.csv")
    print(f"summary_rows={len(summary)}")
    print(summary.groupby("dimension").size().to_string())


if __name__ == "__main__":
    main()
