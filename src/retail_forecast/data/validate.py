"""Validation du schéma et des bornes des données brutes."""

from typing import Any

import pandas as pd


def validate_raw(df: pd.DataFrame, expected: dict[str, Any]) -> dict[str, Any]:
    """Valide le schéma, la cardinalité et la période des données brutes."""
    expected_columns = list(expected["columns"])
    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"Colonnes invalides : attendu {expected_columns}, obtenu {actual_columns}"
        )
    if len(df) != expected["n_rows"]:
        raise ValueError(
            f"Nombre de lignes invalide : attendu {expected['n_rows']}, obtenu {len(df)}"
        )

    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    minimum = pd.Timestamp(expected["date_min"])
    maximum_exclusive = pd.Timestamp(expected["date_max"]) + pd.Timedelta(days=1)
    if dates.isna().any() or (dates < minimum).any() or (dates >= maximum_exclusive).any():
        raise ValueError(
            "InvoiceDate hors plage : attendu "
            f"[{minimum.date()}, {maximum_exclusive.date()})"
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "date_min": dates.min().isoformat(),
        "date_max": dates.max().isoformat(),
        "missing": {column: int(df[column].isna().sum()) for column in df.columns},
    }
