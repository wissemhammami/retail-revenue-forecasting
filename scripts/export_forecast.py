"""Exporte la prévision Prophet figée et son intervalle empirique."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.backtest import run_backtest
from retail_forecast.evaluation.closures import (
    is_known_closed,
    is_reliable_after_closure,
)
from retail_forecast.evaluation.splits import build_folds
from retail_forecast.models.baseline import naive_seasonal_fallback_forecast
from retail_forecast.models.prophet_model import forecast as prophet_forecast
from retail_forecast.models.xgb_model import forecast as xgb_forecast


ROOT = Path(__file__).resolve().parents[1]


def _bucket(value: int, buckets: list[list[int]]) -> str:
    for start, end in buckets:
        if int(start) <= int(value) <= int(end):
            return f"{start}-{end}"
    raise ValueError(f"Horizon hors des tranches configurées: {value}")


def _forecast_function(model: str, params: dict) -> callable:
    if model == "baseline":
        return naive_seasonal_fallback_forecast
    if model == "prophet":
        return lambda frame, origin, horizon, config: prophet_forecast(
            frame, origin, horizon, config, params=params
        )
    if model == "xgboost":
        return lambda frame, origin, horizon, config: xgb_forecast(
            frame, origin, horizon, config, params=params
        )
    raise ValueError(f"Modèle champion inconnu: {model}")


def _calibrate_intervals(
    backtest: pd.DataFrame,
    config: dict,
    buckets: list[list[int]],
    minimum_count: int = 30,
) -> tuple[pd.DataFrame, dict[str, int]]:
    target_dates = pd.to_datetime(backtest["date"])
    closed = is_known_closed(target_dates, config).to_numpy()
    calibration = backtest.loc[~closed].copy()
    calibration["bucket"] = calibration["horizon"].map(
        lambda value: _bucket(value, buckets)
    )
    calibration["residual"] = calibration["actual"] - calibration["prediction"]
    counts = calibration.groupby("bucket", sort=False).size().to_dict()
    expected_buckets = [_bucket(start, buckets) for start, _ in buckets]
    missing = [name for name in expected_buckets if counts.get(name, 0) < minimum_count]
    if missing:
        raise ValueError(
            "Résidus de calibration insuffisants pour: "
            + ", ".join(missing)
        )
    quantiles = (
        calibration.groupby("bucket")["residual"]
        .quantile([0.1, 0.9])
        .unstack()
    )
    quantiles.columns = ["lower_residual", "upper_residual"]
    return quantiles, {name: int(counts.get(name, 0)) for name in expected_buckets}


def _validate_forecast(export: pd.DataFrame, horizon: int) -> None:
    expected_columns = ["date", "forecast", "lower_80", "upper_80", "reliable"]
    if list(export.columns) != expected_columns:
        raise ValueError(f"Colonnes forecast.csv inattendues: {list(export.columns)}")
    if len(export) != horizon:
        raise ValueError(f"Horizon inattendu: {len(export)} lignes, attendu {horizon}")
    if not pd.api.types.is_bool_dtype(export["reliable"]):
        raise ValueError("forecast.csv contient une colonne reliable non booléenne")
    if export["date"].duplicated().any() or not export["date"].is_monotonic_increasing:
        raise ValueError("Les dates de forecast.csv doivent être uniques et chronologiques")
    values = export[["forecast", "lower_80", "upper_80"]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("forecast.csv contient des valeurs manquantes ou non finies")
    if (
        (export["lower_80"] > export["forecast"]).any()
        or (export["forecast"] > export["upper_80"]).any()
    ):
        raise ValueError("Intervalles invalides: lower_80 <= forecast <= upper_80 requis")


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_revenue.csv", parse_dates=["date"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    with (ROOT / "configs" / "best_params.json").open(encoding="utf-8") as handle:
        best_params = json.load(handle)

    origin = daily["date"].max()
    horizon = int(config["evaluation"]["horizon"])
    champion = best_params["champion"]
    params = best_params.get(champion, {}).get("params", {})
    forecast_fn = _forecast_function(champion, params)

    forecast_frame = forecast_fn(daily, origin, horizon, config)
    dev_folds = build_folds(daily, config, "dev")
    dev_backtest = run_backtest(forecast_fn, daily, dev_folds, config)
    buckets = config["evaluation"]["horizon_buckets"]
    quantiles, counts = _calibrate_intervals(dev_backtest, config, buckets)
    calibration = dev_backtest.copy()
    calibration["closed"] = is_known_closed(calibration["date"], config).to_numpy()
    calibration = calibration.loc[~calibration["closed"]]
    calibration["bucket"] = calibration["horizon"].map(
        lambda value: _bucket(value, buckets)
    )
    calibration["residual"] = calibration["actual"] - calibration["prediction"]
    calibration["covered"] = calibration.apply(
        lambda row: quantiles.loc[row["bucket"], "lower_residual"]
        <= row["residual"]
        <= quantiles.loc[row["bucket"], "upper_residual"],
        axis=1,
    )
    coverage = calibration.groupby("bucket", sort=False)["covered"].mean()

    export = forecast_frame.copy()
    export["bucket"] = export["horizon"].map(lambda value: _bucket(value, buckets))
    export = export.join(quantiles, on="bucket")
    export["forecast"] = export["prediction"]
    export["lower_80"] = np.maximum(
        0.0, export["forecast"] + export["lower_residual"]
    )
    export["upper_80"] = np.maximum(
        export["forecast"], export["forecast"] + export["upper_residual"]
    )
    closed_forecast = is_known_closed(export["date"], config).to_numpy()
    export.loc[closed_forecast, ["forecast", "lower_80", "upper_80"]] = 0.0
    export["reliable"] = is_reliable_after_closure(export["date"], config).to_numpy()
    output = export[["date", "forecast", "lower_80", "upper_80", "reliable"]].copy()
    _validate_forecast(output, horizon)
    output.to_csv(ROOT / "dashboard_data" / "forecast.csv", index=False)

    daily.tail(180)[["date", "revenue"]].to_csv(
        ROOT / "dashboard_data" / "history.csv", index=False
    )
    print(f"Champion: {champion}")
    print("Comptages de calibration: " + ", ".join(
        f"{bucket}={count}" for bucket, count in counts.items()
    ))
    print("Couverture diagnostique dev: " + ", ".join(
        f"{bucket}={coverage[bucket]:.4f}" for bucket in coverage.index
    ))
    print(f"Écrit : {ROOT / 'dashboard_data' / 'forecast.csv'} ({len(output)} lignes)")
    print(f"Écrit : {ROOT / 'dashboard_data' / 'history.csv'} (180 lignes)")


if __name__ == "__main__":
    main()
