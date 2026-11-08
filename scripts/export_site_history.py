"""Write docs/data/history.json (last 180 days of revenue) for the static site."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

history = pd.read_csv(ROOT / "dashboard_data" / "history.csv")
out = ROOT / "docs" / "data" / "history.json"
out.parent.mkdir(parents=True, exist_ok=True)
history.to_json(out, orient="records")
print(f"Wrote {len(history)} rows to {out}")
