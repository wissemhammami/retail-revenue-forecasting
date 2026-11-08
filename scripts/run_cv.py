"""Script d'évaluation en validation croisée (CV) et sélection du champion."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

from retail_forecast.data.load import load_config
from retail_forecast.evaluation.backtest import run_backtest
from retail_forecast.evaluation.metrics import (
    bias,
    horizon_bucket,
    mae,
    per_origin_wape,
    rmse,
    summarize,
    wape,
)
from retail_forecast.evaluation.selection import count_better_origins, is_qualified, select_champion
from retail_forecast.evaluation.splits import build_folds
from retail_forecast.models.baseline import naive_seasonal_fallback_forecast
from retail_forecast.models.prophet_model import forecast as prophet_forecast
from retail_forecast.models.xgb_model import forecast as xgb_forecast

ROOT = Path(__file__).resolve().parents[1]


def format_prophet_params_id(params: dict[str, Any]) -> str:
    """Formatte l'identifiant pour Prophet : prophet|cps=0.01|mode=additive."""
    return f"prophet|cps={params['changepoint_prior_scale']}|mode={params['seasonality_mode']}"


def format_xgb_params_id(params: dict[str, Any]) -> str:
    """Formatte l'identifiant pour XGBoost : xgboost|depth=2|n_est=100|lr=0.03."""
    return f"xgboost|depth={params['max_depth']}|n_est={params['n_estimators']}|lr={params['learning_rate']}"


def build_candidates_list(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Construit la liste ordonnée des 15 candidats."""
    candidates: list[dict[str, Any]] = []

    # 1. Baseline
    candidates.append({
        "model": "baseline",
        "params_id": "baseline",
        "params": {},
        "forecast_fn": lambda daily, origin, horizon, cfg, p=None: naive_seasonal_fallback_forecast(
            daily, origin, horizon, cfg
        ),
    })

    # 2. Prophet grid (6 combinaisons)
    p_grid = config["models"]["prophet"]["grid"]
    p_keys = list(p_grid.keys())
    for vals in itertools.product(*(p_grid[k] for k in p_keys)):
        p_dict = dict(zip(p_keys, vals))
        candidates.append({
            "model": "prophet",
            "params_id": format_prophet_params_id(p_dict),
            "params": p_dict,
            "forecast_fn": lambda daily, origin, horizon, cfg, p=p_dict: prophet_forecast(
                daily, origin, horizon, cfg, params=p
            ),
        })

    # 3. XGBoost grid (8 combinaisons)
    x_grid = config["models"]["xgboost"]["grid"]
    x_keys = list(x_grid.keys())
    for vals in itertools.product(*(x_grid[k] for k in x_keys)):
        x_dict = dict(zip(x_keys, vals))
        candidates.append({
            "model": "xgboost",
            "params_id": format_xgb_params_id(x_dict),
            "params": x_dict,
            "forecast_fn": lambda daily, origin, horizon, cfg, p=x_dict: xgb_forecast(
                daily, origin, horizon, cfg, params=p
            ),
        })

    return candidates


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    daily_path = ROOT / "data" / "processed" / "daily_revenue.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {daily_path}")

    daily = pd.read_csv(daily_path, parse_dates=["date"])

    # Construction et validation stricte des folds
    dev_folds = build_folds(daily, config, "dev")
    if len(dev_folds) != 23:
        raise ValueError(f"Attendu 23 folds de dev, obtenu {len(dev_folds)}")
    for f in dev_folds:
        if f["kind"] != "dev":
            raise ValueError(f"Fold non 'dev' détecté: kind={f['kind']}, origin={f['origin']}")

    buckets = config["evaluation"]["horizon_buckets"]
    candidates = build_candidates_list(config)
    if len(candidates) != 15:
        raise ValueError(f"Attendu 15 candidats, obtenu {len(candidates)}")

    print(f"Lancement de la validation croisée sur {len(dev_folds)} folds de développement pour {len(candidates)} candidats...")

    eval_results = []
    per_origin_records = []
    baseline_per_origin_df = None

    for idx, cand in enumerate(candidates, 1):
        m_name = cand["model"]
        p_id = cand["params_id"]
        fn = cand["forecast_fn"]
        print(f"[{idx:02d}/15] Évaluation de {p_id:42s} ... ", end="", flush=True)

        t0 = time.perf_counter()
        bt_df = run_backtest(fn, daily, dev_folds, config)
        elapsed = time.perf_counter() - t0

        if len(bt_df) != 23 * 28:
            raise ValueError(f"Taille de backtest inattendue ({len(bt_df)}) pour {p_id}")

        # Métriques poolées
        pooled_wape = wape(bt_df["actual"], bt_df["prediction"])
        pooled_mae = mae(bt_df["actual"], bt_df["prediction"])
        pooled_rmse = rmse(bt_df["actual"], bt_df["prediction"])
        pooled_bias = bias(bt_df["actual"], bt_df["prediction"])

        # Tranches d'horizon
        bt_df["bucket"] = [horizon_bucket(h, buckets) for h in bt_df["horizon"]]
        b_summary = summarize(bt_df, by="bucket").set_index("bucket")
        wape_1_7 = float(b_summary.loc["1-7", "wape"])
        wape_8_14 = float(b_summary.loc["8-14", "wape"])
        wape_15_28 = float(b_summary.loc["15-28", "wape"])

        # WAPE par origine
        po_df = per_origin_wape(bt_df)
        po_df["model"] = m_name
        po_df["params_id"] = p_id

        for _, row in po_df.iterrows():
            per_origin_records.append({
                "model": m_name,
                "params_id": p_id,
                "origin": pd.Timestamp(row["origin"]).strftime("%Y-%m-%d"),
                "wape": float(row["wape"]),
            })

        if m_name == "baseline":
            # Contrôle de régression
            expected_wape = 0.3620
            expected_mae = 7937.11
            expected_bias = -360.31
            if abs(pooled_wape - expected_wape) > 0.0005:
                raise ValueError(f"Régression WAPE baseline: {pooled_wape:.4f} != {expected_wape}")
            if abs(pooled_mae - expected_mae) > 1.0:
                raise ValueError(f"Régression MAE baseline: {pooled_mae:.2f} != {expected_mae}")
            if abs(pooled_bias - expected_bias) > 1.0:
                raise ValueError(f"Régression Biais baseline: {pooled_bias:.2f} != {expected_bias}")
            baseline_per_origin_df = po_df.copy()
            n_better = 0
        else:
            if baseline_per_origin_df is None:
                raise RuntimeError("La baseline doit être évaluée en premier")
            n_better = count_better_origins(po_df, baseline_per_origin_df)

        print(f"Terminé en {elapsed:5.1f}s | WAPE: {pooled_wape:.4f} | Better origins: {n_better:2d}/23")

        eval_results.append({
            "model": m_name,
            "params_id": p_id,
            "params_json": json.dumps(cand["params"], sort_keys=True),
            "wape": pooled_wape,
            "mae": pooled_mae,
            "rmse": pooled_rmse,
            "bias": pooled_bias,
            "wape_1_7": wape_1_7,
            "wape_8_14": wape_8_14,
            "wape_15_28": wape_15_28,
            "n_origins_better_than_baseline": n_better,
            "raw_params": cand["params"],
            "po_df": po_df,
        })

    # Identification du best_for_model (WAPE poolé le plus bas)
    # Règle : Choix des paramètres d'un modèle = WAPE poolé le plus bas sur sa grille ;
    # en cas d'égalité, la première combinaison dans l'ordre de la grille.
    df_results = pd.DataFrame(eval_results)
    df_results["best_for_model"] = False

    best_prophet_row = None
    best_xgb_row = None

    for m in ["prophet", "xgboost"]:
        sub = df_results[df_results["model"] == m]
        # min() stable grâce à l'ordre conservé
        min_wape = sub["wape"].min()
        best_idx = sub[sub["wape"] == min_wape].index[0]
        df_results.loc[best_idx, "best_for_model"] = True
        if m == "prophet":
            best_prophet_row = df_results.loc[best_idx]
        else:
            best_xgb_row = df_results.loc[best_idx]

    # Qualification des meilleurs challengers face à la baseline
    baseline_row = df_results[df_results["model"] == "baseline"].iloc[0]
    baseline_wape = float(baseline_row["wape"])

    prophet_is_qual = is_qualified(
        float(best_prophet_row["wape"]),
        baseline_wape,
        int(best_prophet_row["n_origins_better_than_baseline"]),
        23,
    )
    xgb_is_qual = is_qualified(
        float(best_xgb_row["wape"]),
        baseline_wape,
        int(best_xgb_row["n_origins_better_than_baseline"]),
        23,
    )

    candidates_for_selection = [
        {"name": "baseline", "pooled_wape": baseline_wape, "qualified": True},
        {"name": "prophet", "pooled_wape": float(best_prophet_row["wape"]), "qualified": prophet_is_qual},
        {"name": "xgboost", "pooled_wape": float(best_xgb_row["wape"]), "qualified": xgb_is_qual},
    ]

    champ_selection = select_champion(candidates_for_selection)
    champion_name = champ_selection["champion_name"]
    rule_applied = champ_selection["rule_applied"]

    # Export cv_results.csv (15 lignes, aucune colonne de durée)
    cols_cv_results = [
        "model",
        "params_id",
        "params_json",
        "wape",
        "mae",
        "rmse",
        "bias",
        "wape_1_7",
        "wape_8_14",
        "wape_15_28",
        "n_origins_better_than_baseline",
        "best_for_model",
    ]
    cv_results_export = df_results[cols_cv_results].copy()
    cv_results_path = ROOT / "data" / "processed" / "cv_results.csv"
    cv_results_export.to_csv(cv_results_path, index=False)
    print(f"Écrit : {cv_results_path} ({len(cv_results_export)} lignes)")

    # Export cv_per_origin.csv (15 x 23 = 345 lignes)
    df_per_origin = pd.DataFrame(per_origin_records)[["model", "params_id", "origin", "wape"]]
    cv_per_origin_path = ROOT / "data" / "processed" / "cv_per_origin.csv"
    df_per_origin.to_csv(cv_per_origin_path, index=False)
    print(f"Écrit : {cv_per_origin_path} ({len(df_per_origin)} lignes)")

    # Export configs/best_params.json
    best_params_json_content = {
        "rule": (
            "Prophet ou XGBoost ne remplace la baseline que s'il a un WAPE moyen inférieur à la baseline "
            "ET un WAPE par origine inférieur sur la majorité des origines (>= 12 sur 23). "
            "Entre deux challengers qualifiés, le WAPE moyen le plus bas l'emporte. "
            "Si aucun n'est qualifié, la baseline est championne ('une règle simple suffit')."
        ),
        "rule_applied": rule_applied,
        "champion": champion_name,
        "baseline": {
            "model": "baseline",
            "params_id": "baseline",
            "wape": float(baseline_row["wape"]),
            "mae": float(baseline_row["mae"]),
            "rmse": float(baseline_row["rmse"]),
            "bias": float(baseline_row["bias"]),
            "wape_1_7": float(baseline_row["wape_1_7"]),
            "wape_8_14": float(baseline_row["wape_8_14"]),
            "wape_15_28": float(baseline_row["wape_15_28"]),
        },
        "prophet": {
            "model": "prophet",
            "params_id": str(best_prophet_row["params_id"]),
            "params": best_prophet_row["raw_params"],
            "wape": float(best_prophet_row["wape"]),
            "mae": float(best_prophet_row["mae"]),
            "rmse": float(best_prophet_row["rmse"]),
            "bias": float(best_prophet_row["bias"]),
            "wape_1_7": float(best_prophet_row["wape_1_7"]),
            "wape_8_14": float(best_prophet_row["wape_8_14"]),
            "wape_15_28": float(best_prophet_row["wape_15_28"]),
            "n_origins_better_than_baseline": int(best_prophet_row["n_origins_better_than_baseline"]),
            "qualified": bool(prophet_is_qual),
            "qualification_reason": (
                f"WAPE {best_prophet_row['wape']:.4f} {'<' if best_prophet_row['wape'] < baseline_wape else '>='} {baseline_wape:.4f} "
                f"et {best_prophet_row['n_origins_better_than_baseline']}/23 origines meilleures (seuil: 12)"
            ),
        },
        "xgboost": {
            "model": "xgboost",
            "params_id": str(best_xgb_row["params_id"]),
            "params": best_xgb_row["raw_params"],
            "wape": float(best_xgb_row["wape"]),
            "mae": float(best_xgb_row["mae"]),
            "rmse": float(best_xgb_row["rmse"]),
            "bias": float(best_xgb_row["bias"]),
            "wape_1_7": float(best_xgb_row["wape_1_7"]),
            "wape_8_14": float(best_xgb_row["wape_8_14"]),
            "wape_15_28": float(best_xgb_row["wape_15_28"]),
            "n_origins_better_than_baseline": int(best_xgb_row["n_origins_better_than_baseline"]),
            "qualified": bool(xgb_is_qual),
            "qualification_reason": (
                f"WAPE {best_xgb_row['wape']:.4f} {'<' if best_xgb_row['wape'] < baseline_wape else '>='} {baseline_wape:.4f} "
                f"et {best_xgb_row['n_origins_better_than_baseline']}/23 origines meilleures (seuil: 12)"
            ),
        },
    }

    best_params_path = ROOT / "configs" / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_json_content, f, indent=2, ensure_ascii=False)
    print(f"Écrit : {best_params_path}")


if __name__ == "__main__":
    main()
