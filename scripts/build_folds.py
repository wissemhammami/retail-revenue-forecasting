"""Write development and test folds."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.splits import build_folds, overlap_summary


def main() -> None:
    """Build and display the time split."""
    config = load_config()
    processed = Path(config["paths"]["processed"])
    daily = pd.read_csv(processed / "daily_revenue.csv", parse_dates=["date"])
    folds = build_folds(daily, config, "dev") + build_folds(daily, config, "test")
    output = pd.DataFrame(folds)
    temporary = processed / "folds.csv.part"
    output.to_csv(temporary, index=False)
    temporary.replace(processed / "folds.csv")
    for kind in ("dev", "test"):
        subset = [fold for fold in folds if fold["kind"] == kind]
        print(f"{kind}_folds={len(subset)}")
        print(pd.Series([fold["regime"] for fold in subset]).value_counts().to_string())
        print(pd.DataFrame(subset).head(5).to_string(index=False))
        print(pd.DataFrame(subset).tail(5).to_string(index=False))
        print(f"{kind}_overlap={overlap_summary(subset)}")


if __name__ == "__main__":
    main()
