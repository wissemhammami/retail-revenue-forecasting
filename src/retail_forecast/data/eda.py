"""Fonctions d'analyse exploratoire sans effets graphiques."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult

from retail_forecast.data.clean import (
    add_revenue,
    drop_cancelled_orders,
    drop_exact_duplicates,
    flag_cancellations,
    remove_invalid_prices,
    remove_invalid_quantities,
    remove_non_product_codes,
    remove_non_product_invoices,
    winsorize,
)


PARTIAL_DAY = pd.Timestamp("2011-12-09")
WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _without_partial(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Retire le jour partiel de la fin de la source."""
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])
    return result.loc[dates.dt.normalize().ne(PARTIAL_DAY)].copy()


def eda_summary(clean: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Construit un résumé long par jour, mois, pays et produit."""
    transactions = _without_partial(clean, "InvoiceDate")
    transactions["weekday"] = pd.to_datetime(transactions["InvoiceDate"]).dt.dayofweek.map(
        WEEKDAY_NAMES
    )
    transactions["month"] = pd.to_datetime(
        transactions["InvoiceDate"]
    ).dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []

    daily_work = _without_partial(daily, "date")
    daily_work["weekday"] = pd.to_datetime(daily_work["date"]).dt.dayofweek.map(
        WEEKDAY_NAMES
    )
    daily_work = daily_work.loc[daily_work["is_trading_day"].astype(bool)]
    transaction_counts = transactions.groupby("weekday").size()
    daily_means = daily_work.groupby("weekday")["revenue"].mean()
    for key in WEEKDAY_NAMES.values():
        if key in daily_means.index:
            rows.append(
                {
                    "dimension": "weekday",
                    "key": key,
                    "revenue": daily_means[key],
                    "n_transactions": int(transaction_counts.get(key, 0)),
                }
            )

    for key, group in transactions.groupby("month", sort=True):
        rows.append(
            {
                "dimension": "month",
                "key": key,
                "revenue": group["Revenue"].sum(),
                "n_transactions": len(group),
            }
        )

    for key, group in transactions.groupby("Country", sort=True):
        rows.append(
            {
                "dimension": "country",
                "key": str(key),
                "revenue": group["Revenue"].sum(),
                "n_transactions": len(group),
            }
        )

    product_groups = transactions.assign(
        StockCode=transactions["StockCode"].astype("string"),
        Description=transactions["Description"].fillna("").astype("string"),
    )
    product_totals = (
        product_groups.groupby("StockCode", dropna=False)
        .agg(revenue=("Revenue", "sum"), n_transactions=("Revenue", "size"))
        .sort_values("revenue", ascending=False)
        .head(20)
    )
    descriptions = (
        product_groups.groupby(["StockCode", "Description"], dropna=False)
        .size()
        .rename("frequency")
        .reset_index()
        .sort_values(["StockCode", "frequency"], ascending=[True, False])
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    for stock_code, row in product_totals.iterrows():
        rows.append(
            {
                "dimension": "product",
                "key": f"{stock_code} | {descriptions.get(stock_code, '')}",
                "revenue": row["revenue"],
                "n_transactions": row["n_transactions"],
            }
        )

    return pd.DataFrame(
        rows, columns=["dimension", "key", "revenue", "n_transactions"]
    )


def weekday_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Calcule moyenne, médiane et nombre de jours actifs par weekday."""
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.loc[work["is_trading_day"].astype(bool)]
    work["weekday"] = work["date"].dt.dayofweek.map(WEEKDAY_NAMES)
    result = (
        work.groupby(["weekday", "date"], as_index=False)["revenue"].sum()
        .groupby("weekday", sort=False)["revenue"]
        .agg(mean="mean", median="median", n_days="count")
        .reset_index()
    )
    return result


def decompose_weekly(daily: pd.DataFrame) -> DecomposeResult:
    """Décompose les jours actifs sans interpoler les jours fermés.

    Les jours fermés et la fenêtre 2010-12-24 à 2011-01-03 sont retirés.
    La saisonnalité est le profil moyen par jour de semaine et le résidu est
    calculé autour d'une tendance roulante de sept observations actives.
    """
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.loc[
        work["is_trading_day"].astype(bool)
        & ~work["date"].between("2010-12-24", "2011-01-03")
    ].copy()
    work["weekday"] = work["date"].dt.dayofweek
    profile = work.groupby("weekday")["revenue"].transform("mean")
    trend = work["revenue"].rolling(7, center=True, min_periods=1).mean()
    observed = pd.Series(work["revenue"].to_numpy(), index=work["date"], name="observed")
    seasonal = pd.Series(profile.to_numpy(), index=work["date"], name="seasonal")
    trend_series = pd.Series(trend.to_numpy(), index=work["date"], name="trend")
    resid = pd.Series(
        (observed - trend_series - seasonal + seasonal.mean()).to_numpy(),
        index=work["date"],
        name="resid",
    )
    return DecomposeResult(observed, trend_series, seasonal, resid)


def anomaly_details(
    clean: pd.DataFrame, anomalous_days: pd.DataFrame
) -> pd.DataFrame:
    """Retourne les trois plus grosses transactions de chaque jour anormal."""
    transactions = _without_partial(clean, "InvoiceDate")
    transactions["date"] = pd.to_datetime(transactions["InvoiceDate"]).dt.normalize()
    anomaly_dates = pd.to_datetime(anomalous_days["date"]).dt.normalize()
    selected = transactions.loc[transactions["date"].isin(anomaly_dates)].copy()
    selected["day_revenue"] = selected.groupby("date")["Revenue"].transform("sum")
    selected["share_of_day"] = selected["Revenue"] / selected["day_revenue"]
    selected = selected.sort_values(["date", "Revenue"], ascending=[True, False])
    selected["rank"] = selected.groupby("date").cumcount() + 1
    return selected.loc[
        selected["rank"] <= 3,
        [
            "date",
            "rank",
            "InvoiceNo",
            "StockCode",
            "CustomerID",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "share_of_day",
        ],
    ].reset_index(drop=True)


def cleaning_impact(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Compare trois périmètres homogènes de lignes et de revenu."""
    raw_work = raw.copy()
    raw_work["Revenue"] = raw_work["Quantity"] * raw_work["UnitPrice"]
    clean_work = clean.copy()
    return pd.DataFrame(
        [
            {
                "dataset": "raw",
                "rows": len(raw_work),
                "revenue": raw_work["Revenue"].sum(),
            },
            {
                "dataset": "raw_without_cancellations",
                "rows": int(
                    (~raw_work["InvoiceNo"].astype("string").str.startswith("C", na=False)).sum()
                ),
                "revenue": raw_work.loc[
                    ~raw_work["InvoiceNo"].astype("string").str.startswith("C", na=False),
                    "Revenue",
                ].sum(),
            },
            {
                "dataset": "clean",
                "rows": len(clean_work),
                "revenue": clean_work["Revenue"].sum(),
            },
        ]
    )


def cleaning_impact_steps(
    raw: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calcule les lignes et revenus retirés à chaque étape du nettoyage."""
    current = raw.copy()
    cleaning = config["cleaning"]
    rows: list[dict[str, Any]] = []
    operations = [
        ("drop_exact_duplicates", lambda frame: drop_exact_duplicates(frame)),
        ("flag_cancellations", lambda frame: flag_cancellations(frame)),
        (
            "drop_cancelled_orders",
            lambda frame: drop_cancelled_orders(frame),
        ),
        (
            "exclude_cancellations",
            lambda frame: flag_cancellations(frame).loc[
                ~flag_cancellations(frame)["is_cancellation"]
            ].copy(),
        ),
        ("remove_invalid_prices", remove_invalid_prices),
        (
            "remove_non_product_codes",
            lambda frame: remove_non_product_codes(
                frame,
                cleaning["non_product_codes"],
                cleaning["non_product_prefixes"],
            ),
        ),
        (
            "remove_non_product_invoices",
            lambda frame: remove_non_product_invoices(
                frame, cleaning.get("non_product_invoice_prefixes", [])
            ),
        ),
        (
            "remove_invalid_quantities",
            lambda frame: remove_invalid_quantities(
                frame, cleaning["returns_mode"]
            ),
        ),
        ("winsorize", lambda frame: winsorize(frame, cleaning)),
        ("add_revenue", add_revenue),
    ]
    requested_steps = set(cleaning_summary["step"])
    for step, operation in operations:
        if step not in requested_steps:
            continue
        if step == "drop_cancelled_orders" and not cleaning["drop_cancelled_orders"]:
            continue
        before = current.copy()
        if "Revenue" not in before:
            before["Revenue"] = before["Quantity"] * before["UnitPrice"]
        current = operation(current)
        before_revenue = before["Revenue"].sum()
        after_revenue = (
            current["Revenue"].sum()
            if "Revenue" in current
            else (current["Quantity"] * current["UnitPrice"]).sum()
        )
        rows.append(
            {
                "step": step,
                "rows_removed": len(before) - len(current),
                "revenue_removed": before_revenue - after_revenue,
            }
        )
    return pd.DataFrame(rows)
