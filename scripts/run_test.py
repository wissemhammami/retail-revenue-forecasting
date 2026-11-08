"""Évalue les trois modèles figés sur les folds de test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.backtest import run_backtest
from retail_forecast.evaluation.metrics import bias, horizon_bucket, mae, per_origin_wape, rmse, wape
from retail_forecast.evaluation.splits import build_folds
from retail_forecast.models.baseline import naive_seasonal_fallback_forecast
from retail_forecast.models.prophet_model import forecast as prophet_forecast
from retail_forecast.models.xgb_model import forecast as xgb_forecast


ROOT = Path(__file__).resolve().parents[1]


def _metrics(frame: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}wape": wape(frame["actual"], frame["prediction"]),
        f"{prefix}mae": mae(frame["actual"], frame["prediction"]),
        f"{prefix}rmse": rmse(frame["actual"], frame["prediction"]),
        f"{prefix}bias": bias(frame["actual"], frame["prediction"]),
    }


def main() -> None:
    test_results_path = ROOT / "data" / "processed" / "test_results.csv"
    if test_results_path.exists():
        raise FileExistsError(f"Refus d'écraser le fichier existant: {test_results_path}")

    config = load_config(ROOT / "configs" / "config.yaml")
    daily_path = ROOT / "data" / "processed" / "daily_revenue.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    with (ROOT / "configs" / "best_params.json").open(encoding="utf-8") as handle:
        best_params: dict[str, Any] = json.load(handle)

    test_folds = build_folds(daily, config, "test")
    if len(test_folds) != 11:
        raise ValueError(f"Attendu 11 folds de test, obtenu {len(test_folds)}")
    if any(fold["kind"] != "test" for fold in test_folds):
        raise ValueError("Tous les folds doivent être de kind 'test'")

    models: list[tuple[str, Callable[..., pd.DataFrame]]] = [
        (
            "baseline",
            lambda frame, origin, horizon, cfg: naive_seasonal_fallback_forecast(
                frame, origin, horizon, cfg
            ),
        ),
        (
            "prophet",
            lambda frame, origin, horizon, cfg: prophet_forecast(
                frame,
                origin,
                horizon,
                cfg,
                params=best_params["prophet"]["params"],
            ),
        ),
        (
            "xgboost",
            lambda frame, origin, horizon, cfg: xgb_forecast(
                frame,
                origin,
                horizon,
                cfg,
                params=best_params["xgboost"]["params"],
            ),
        ),
    ]

    buckets = config["evaluation"]["horizon_buckets"]
    result_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    backtests: dict[str, pd.DataFrame] = {}

    for model_name, forecast_fn in models:
        backtest = run_backtest(forecast_fn, daily, test_folds, config)
        backtests[model_name] = backtest
        row: dict[str, Any] = {"model": model_name, **_metrics(backtest)}
        backtest = backtest.copy()
        backtest["bucket"] = [
            horizon_bucket(horizon, buckets) for horizon in backtest["horizon"]
        ]
        for bucket, bucket_frame in backtest.groupby("bucket", sort=False):
            row.update(_metrics(bucket_frame, prefix=f"{bucket}_"))
        result_rows.append(row)

        per_origin = per_origin_wape(backtest)
        regimes = backtest[["origin", "regime"]].drop_duplicates()
        per_origin = per_origin.merge(regimes, on="origin", how="left")
        origin_rows.extend(
            {
                "model": model_name,
                "origin": pd.Timestamp(record["origin"]).strftime("%Y-%m-%d"),
                "regime": record["regime"],
                "wape": float(record["wape"]),
            }
            for record in per_origin.to_dict("records")
        )

    pd.DataFrame(result_rows).to_csv(test_results_path, index=False)
    test_per_origin_path = ROOT / "data" / "processed" / "test_per_origin.csv"
    pd.DataFrame(origin_rows).to_csv(test_per_origin_path, index=False)
    detailed = next(iter(backtests.values()))[["origin", "date", "horizon", "actual"]].copy()
    for model_name, backtest in backtests.items():
        detailed[model_name if model_name != "xgboost" else "xgb"] = (
            backtest["prediction"].to_numpy()
        )
    detailed_path = ROOT / "data" / "processed" / "predictions_detailed.csv"
    detailed.to_csv(detailed_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    for model_name, backtest in backtests.items():
        bucketed = backtest.assign(
            horizon_bucket=[
                horizon_bucket(value, buckets) for value in backtest["horizon"]
            ]
        )
        for bucket, bucket_frame in bucketed.groupby("horizon_bucket", sort=False):
            metric_rows.append(
                {
                    "model": model_name,
                    "horizon_bucket": bucket,
                    "MAE": mae(bucket_frame["actual"], bucket_frame["prediction"]),
                    "RMSE": rmse(bucket_frame["actual"], bucket_frame["prediction"]),
                    "WAPE": wape(bucket_frame["actual"], bucket_frame["prediction"]),
                    "bias": bias(bucket_frame["actual"], bucket_frame["prediction"]),
                }
            )
    metrics_path = ROOT / "dashboard_data" / "metrics.csv"
    pd.DataFrame(
        metric_rows,
        columns=["model", "horizon_bucket", "MAE", "RMSE", "WAPE", "bias"],
    ).to_csv(metrics_path, index=False)

    print(f"Écrit : {test_results_path} ({len(result_rows)} lignes)")
    print(f"Écrit : {test_per_origin_path} ({len(origin_rows)} lignes)")
    print(f"Écrit : {detailed_path} ({len(detailed)} lignes)")
    print(f"Écrit : {metrics_path} ({len(metric_rows)} lignes)")
    print("WAPE poolé par régime")
    for regime in ("normal", "montee_novembre", "decembre"):
        print(f"régime={regime}")
        for model_name, backtest in backtests.items():
            regime_frame = backtest[backtest["regime"] == regime]
            print(f"  {model_name}: {wape(regime_frame['actual'], regime_frame['prediction']):.10f}")

    baseline_wape = (
        pd.DataFrame(origin_rows).query("model == 'baseline'").set_index("origin")["wape"]
    )
    print("Origines strictement meilleures que la baseline")
    for model_name in ("prophet", "xgboost"):
        challenger_wape = (
            pd.DataFrame(origin_rows).query(f"model == '{model_name}'").set_index("origin")["wape"]
        )
        better = int((challenger_wape < baseline_wape).sum())
        print(f"{model_name}: {better}/11")

    prophet_pool = result_rows[1]["wape"]
    baseline_pool = result_rows[0]["wape"]
    prophet_better = int(
        (
            pd.DataFrame(origin_rows).query("model == 'prophet'").set_index("origin")["wape"]
            < baseline_wape
        ).sum()
    )
    confirmed = prophet_pool < baseline_pool and prophet_better >= 6
    print(
        "Verdict Prophet: "
        + ("confirmé sur test" if confirmed else "non confirmé sur test")
    )


if __name__ == "__main__":
    main()
