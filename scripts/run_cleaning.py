"""Exécute le nettoyage et écrit les artefacts intermédiaires."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_forecast.data.clean import (
    drop_cancelled_orders,
    drop_exact_duplicates,
    flag_cancellations,
    remove_invalid_prices,
    remove_invalid_quantities,
    remove_non_product_codes,
    remove_non_product_invoices,
    add_revenue,
    clean_pipeline,
    winsorize,
)
from retail_forecast.data.load import load_config, read_raw


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Écrit un CSV par fichier temporaire puis remplacement atomique."""
    temporary = destination.with_name(destination.name + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    """Nettoie le CSV brut et affiche le journal d'audit."""
    config = load_config()
    raw = read_raw(config)
    cleaned, summary = clean_pipeline(raw, config)
    interim = Path(config["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(cleaned, interim / "clean_transactions.csv")
    summary_frame = pd.DataFrame(summary)
    atomic_write_csv(summary_frame, interim / "cleaning_summary.csv")

    cleaning = config["cleaning"]
    before_winsor = drop_exact_duplicates(raw)
    before_winsor = flag_cancellations(before_winsor)
    if cleaning["drop_cancelled_orders"]:
        before_winsor = drop_cancelled_orders(before_winsor)
    before_winsor = flag_cancellations(before_winsor)
    if cleaning["returns_mode"] == "exclude":
        before_winsor = before_winsor.loc[
            ~before_winsor["is_cancellation"]
        ].copy()
    before_winsor = remove_invalid_prices(before_winsor)
    before_winsor = remove_non_product_codes(
        before_winsor,
        cleaning["non_product_codes"],
        cleaning["non_product_prefixes"],
    )
    before_winsor = remove_non_product_invoices(
        before_winsor, cleaning.get("non_product_invoice_prefixes", [])
    )
    before_winsor = remove_invalid_quantities(
        before_winsor, cleaning["returns_mode"]
    )
    before_winsor = before_winsor.drop(columns=["is_cancellation"])
    quantity_capped = winsorize(before_winsor, cleaning)
    after_winsor = add_revenue(quantity_capped)
    effects = pd.DataFrame(
        [
            {
                "stage": "before_winsorization",
                "max_quantity": before_winsor["Quantity"].max(),
                "max_unit_price": before_winsor["UnitPrice"].max(),
                "max_revenue": (before_winsor["Quantity"] * before_winsor["UnitPrice"]).max(),
                "total_revenue": (before_winsor["Quantity"] * before_winsor["UnitPrice"]).sum(),
                "quantity_capped": 0,
                "price_capped": 0,
            },
            {
                "stage": "after_winsorization",
                "max_quantity": cleaned["Quantity"].max(),
                "max_unit_price": cleaned["UnitPrice"].max(),
                "max_revenue": cleaned["Revenue"].max(),
                "total_revenue": cleaned["Revenue"].sum(),
                "quantity_capped": int(cleaned["quantity_capped"].sum()),
                "price_capped": int(cleaned["price_capped"].sum()),
            },
        ]
    )
    atomic_write_csv(effects, interim / "cleaning_effects.csv")
    print(summary_frame.to_string(index=False))
    print(f"final_rows={len(cleaned)}")


if __name__ == "__main__":
    main()
