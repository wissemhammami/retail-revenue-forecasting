"""Write historical features."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.splits import build_folds
from retail_forecast.features.build_features import build_features


def atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    """Write a CSV and rename it atomically."""
    temporary = destination.with_name(destination.name + ".part")
    frame.to_csv(temporary, index=True, index_label="date")
    temporary.replace(destination)


def main() -> None:
    """Build features.csv from the daily series."""
    config = load_config()
    processed = Path(config["paths"]["processed"])
    daily = pd.read_csv(processed / "daily_revenue.csv", parse_dates=["date"])
    features = build_features(daily, config)
    processed.mkdir(parents=True, exist_ok=True)
    atomic_write(features, processed / "features.csv")
    folds = build_folds(daily, config, "dev") + build_folds(daily, config, "test")
    atomic_write(pd.DataFrame(folds).drop(columns=["train_days", "overlap_days_previous"]),
                 processed / "folds.csv")
    print(f"feature_rows={len(features)}")
    print(f"feature_columns={list(features.columns)}")
    print(features.isna().sum().to_string())
    print(f"folds={len(folds)}")


if __name__ == "__main__":
    main()
