"""Clean transactions before any aggregation."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd


LOGGER = logging.getLogger(__name__)


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove strictly duplicated rows."""
    return df.drop_duplicates().copy()


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean indicating invoices whose number starts with C."""
    result = df.copy()
    result["is_cancellation"] = result["InvoiceNo"].astype("string").str.startswith(
        "C", na=False
    )
    return result


def drop_cancelled_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Remove one matched prior order for each cancellation.

    Matching requires CustomerID, StockCode, and opposite quantity. Each
    cancellation removes one prior order, the most recent one. Without
    CustomerID, no matching is performed.
    """
    result = flag_cancellations(df)
    result["_source_order"] = range(len(result))
    to_drop: set[int] = set()
    candidates: dict[tuple[Any, Any, Any], list[tuple[pd.Timestamp, int]]] = {}
    for _, order in result.loc[~result["is_cancellation"]].iterrows():
        customer_id = order["CustomerID"]
        stock_code = order["StockCode"]
        quantity = order["Quantity"]
        invoice_date = order["InvoiceDate"]
        if (
            pd.isna(customer_id)
            or pd.isna(stock_code)
            or pd.isna(quantity)
            or pd.isna(invoice_date)
        ):
            continue
        key = (customer_id, stock_code, quantity)
        candidates.setdefault(key, []).append(
            (invoice_date, int(order["_source_order"]))
        )
    for matches in candidates.values():
        matches.sort()

    cancellations = result.loc[result["is_cancellation"]].sort_values(
        ["InvoiceDate", "_source_order"]
    )

    for cancellation_index, cancellation in cancellations.iterrows():
        customer_id = cancellation["CustomerID"]
        stock_code = cancellation["StockCode"]
        quantity = cancellation["Quantity"]
        if pd.isna(customer_id) or pd.isna(stock_code) or pd.isna(quantity):
            continue
        key = (customer_id, stock_code, -quantity)
        for invoice_date, source_order in reversed(candidates.get(key, [])):
            if invoice_date < cancellation["InvoiceDate"] and source_order not in to_drop:
                to_drop.add(source_order)
                break

    result = result.loc[~result["_source_order"].isin(to_drop)].drop(
        columns=["is_cancellation", "_source_order"]
    )
    return result


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only transactions whose UnitPrice is strictly positive."""
    return df.loc[df["UnitPrice"].gt(0)].copy()


def remove_non_product_codes(
    df: pd.DataFrame, codes: list[str], prefixes: list[str]
) -> pd.DataFrame:
    """Exclude non-product codes and prefixes, ignoring case."""
    stock_codes = df["StockCode"].astype("string").str.upper()
    excluded_codes = {code.upper() for code in codes}
    excluded_prefixes = tuple(prefix.upper() for prefix in prefixes)
    excluded = stock_codes.isin(excluded_codes)
    if excluded_prefixes:
        excluded |= stock_codes.str.startswith(excluded_prefixes, na=False)
    return df.loc[~excluded].copy()


def remove_non_product_invoices(
    df: pd.DataFrame, prefixes: list[str]
) -> pd.DataFrame:
    """Exclude invoices whose number starts with a configured prefix."""
    invoice_numbers = df["InvoiceNo"].astype("string").str.upper()
    excluded_prefixes = tuple(prefix.upper() for prefix in prefixes)
    if not excluded_prefixes:
        return df.copy()
    return df.loc[~invoice_numbers.str.startswith(excluded_prefixes, na=False)].copy()


def remove_invalid_quantities(
    df: pd.DataFrame, returns_mode: str = "exclude"
) -> pd.DataFrame:
    """Exclude non-positive quantities, except returns retained in net mode."""
    if returns_mode not in {"exclude", "net"}:
        raise ValueError("returns_mode must be 'exclude' or 'net'")
    if returns_mode == "net":
        return df.loc[df["Quantity"].ne(0)].copy()
    return df.loc[df["Quantity"].gt(0)].copy()


def winsorize(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Cap Quantity and UnitPrice and mark modified rows."""
    result = df.copy()
    result["quantity_capped"] = False
    result["price_capped"] = False
    settings = config["winsorize"]
    if not settings["enabled"] or result.empty:
        return result

    result["Quantity"] = result["Quantity"].astype(float)
    quantity_limit = result["Quantity"].quantile(
        settings["quantity_upper_quantile"]
    )
    price_limit = result["UnitPrice"].quantile(
        settings["unit_price_upper_quantile"]
    )
    result["quantity_capped"] = result["Quantity"].gt(quantity_limit)
    result["price_capped"] = result["UnitPrice"].gt(price_limit)
    result.loc[result["quantity_capped"], "Quantity"] = quantity_limit
    result.loc[result["price_capped"], "UnitPrice"] = price_limit
    LOGGER.info(
        "Winsorization: %d Quantity rows, %d UnitPrice rows",
        int(result["quantity_capped"].sum()),
        int(result["price_capped"].sum()),
    )
    return result


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Revenue after any capping."""
    result = df.copy()
    result["Revenue"] = result["Quantity"] * result["UnitPrice"]
    return result


def clean_pipeline(
    df: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply cleaning rules in the specified order."""
    cleaning = config["cleaning"]
    current = df.copy()
    summary: list[dict[str, Any]] = []

    def record(
        step: str, before: int, after: int, note: str = ""
    ) -> None:
        summary.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "removed": before - after,
                "note": note,
            }
        )

    before = len(current)
    current = drop_exact_duplicates(current)
    record("drop_exact_duplicates", before, len(current))

    before = len(current)
    current = flag_cancellations(current)
    record("flag_cancellations", before, len(current))

    if cleaning["drop_cancelled_orders"]:
        before = len(current)
        current = drop_cancelled_orders(current)
        record("drop_cancelled_orders", before, len(current))

    current = flag_cancellations(current)
    if cleaning["returns_mode"] == "exclude":
        before = len(current)
        current = current.loc[~current["is_cancellation"]].copy()
        record("exclude_cancellations", before, len(current))

    before = len(current)
    current = remove_invalid_prices(current)
    record("remove_invalid_prices", before, len(current))

    before = len(current)
    current = remove_non_product_codes(
        current,
        cleaning["non_product_codes"],
        cleaning["non_product_prefixes"],
    )
    record("remove_non_product_codes", before, len(current))

    before = len(current)
    current = remove_non_product_invoices(
        current, cleaning.get("non_product_invoice_prefixes", [])
    )
    record("remove_non_product_invoices", before, len(current))

    before = len(current)
    current = remove_invalid_quantities(current, cleaning["returns_mode"])
    record("remove_invalid_quantities", before, len(current))

    before = len(current)
    current = winsorize(current, cleaning)
    capped = int(current["quantity_capped"].sum() + current["price_capped"].sum())
    record("winsorize", before, len(current), f"{capped} caps")

    before = len(current)
    current = add_revenue(current)
    record("add_revenue", before, len(current), "0 rows removed")
    if cleaning["winsorize"]["enabled"]:
        before = len(current)
        current = drop_exact_duplicates(current)
        record(
            "drop_exact_duplicates_after_winsorization",
            before,
            len(current),
            "duplicates created by capping",
        )
    current = current.drop(columns=["is_cancellation"])
    return current, summary
