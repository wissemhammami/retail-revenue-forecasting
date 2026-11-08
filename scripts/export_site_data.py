"""Export reliable forecast rows for the static site."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write reliable forecast rows and export metadata."""
    source = ROOT / "dashboard_data" / "forecast.csv"
    if not source.exists():
        raise FileNotFoundError(f"Forecast export absent: {source}")

    forecast = pd.read_csv(source, parse_dates=["date"])
    required = {"date", "forecast", "lower_80", "upper_80", "reliable"}
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"Forecast export columns missing: {sorted(missing)}")

    reliable = forecast.loc[forecast["reliable"]].drop(columns=["reliable"])
    output_dir = ROOT / "docs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    reliable.to_json(output_dir / "forecast.json", orient="records", date_format="iso")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {"forecast_days_dropped": int((~forecast["reliable"]).sum())},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'forecast.json'} ({len(reliable)} rows)")
    print(f"Wrote {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
