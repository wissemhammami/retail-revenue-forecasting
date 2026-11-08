"""Chargement de la configuration et des données brutes."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge la configuration YAML et résout les chemins du projet."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    config_path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key, value in config["paths"].items():
        config["paths"][key] = str((PROJECT_ROOT / value).resolve())
    return config


def read_raw(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Lit le CSV brut avec les types identifiants et les dates attendus."""
    config = config or load_config()
    raw_path = Path(config["paths"]["raw"]) / config["raw_filename"]
    return pd.read_csv(
        raw_path,
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
        parse_dates=["InvoiceDate"],
    )
