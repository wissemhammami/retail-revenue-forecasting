"""Acquire and validate the UCI Online Retail dataset."""

from pathlib import Path
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.load import load_config
from retail_forecast.data.validate import validate_raw


def download_xlsx(url: str, destination: Path) -> None:
    """Download a file with retries and atomic replacement."""
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, timeout=60, stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            temporary.replace(destination)
            print(f"Downloaded size: {destination.stat().st_size} bytes")
            return
        except (OSError, requests.RequestException) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 3:
                time.sleep(1)
    raise RuntimeError(
        f"Download failed after 3 attempts: {last_error}"
    ) from last_error


def main() -> None:
    """Build the raw CSV without rewriting an already valid CSV."""
    config = load_config()
    raw_dir = Path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / config["raw_filename"]
    expected = config["expected"]

    if csv_path.exists():
        try:
            summary = validate_raw(
                pd.read_csv(
                    csv_path,
                    dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
                    parse_dates=["InvoiceDate"],
                ),
                expected,
            )
            print("Existing CSV is valid; no rewrite needed.")
            print(summary)
            return
        except (OSError, ValueError, pd.errors.ParserError) as error:
            backup = csv_path.with_name(csv_path.name + ".bak")
            csv_path.replace(backup)
            print(f"Invalid CSV renamed to {backup}: {error}")

    xlsx_path = raw_dir / config["source"]["xlsx_filename"]
    if not xlsx_path.exists():
        try:
            download_xlsx(config["source"]["url"], xlsx_path)
        except RuntimeError as error:
            raise SystemExit(
                f"{error}\nDownload '{xlsx_path.name}' from "
                f"{config['source']['url']} and place it in '{raw_dir}'."
            ) from error

    frame = pd.read_excel(xlsx_path, engine="openpyxl")
    summary = validate_raw(frame, expected)
    temporary = csv_path.with_name(csv_path.name + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    print(summary)
    print(f"CSV created: {csv_path}")


if __name__ == "__main__":
    main()
