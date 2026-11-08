"""Minimal presentation dashboard for precomputed retail revenue forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard_data"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
REQUIRED_FORECAST = {"date", "forecast", "lower_80", "upper_80", "reliable"}
REQUIRED_HISTORY = {"date", "revenue"}
REQUIRED_METRICS = {"model", "wape", "mae", "rmse", "bias"}
REGIME_LABELS = {
    "montee_novembre": "November ramp-up",
    "november_ramp": "November ramp-up",
    "decembre": "December",
    "december": "December",
    "normal": "Normal",
}


def _read_csv(name: str, required: set[str], dates: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing export: {path}")
    frame = pd.read_csv(path, parse_dates=dates or [])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
    return frame


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_forecast_data(
    forecast: pd.DataFrame, history: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = int(config["evaluation"]["horizon"])
    closure_dates = set(pd.to_datetime(
        config["aggregation"].get("closure_dates", [])
    ).normalize())
    for frame, label in ((forecast, "forecast"), (history, "history")):
        if frame["date"].isna().any():
            raise ValueError(f"{label} contains invalid dates")
        if frame["date"].duplicated().any():
            raise ValueError(f"{label} contains duplicate dates")
        if not frame["date"].is_monotonic_increasing:
            raise ValueError(f"{label} must be chronological")
    if len(forecast) != horizon:
        raise ValueError(f"Unexpected horizon: {len(forecast)} rows, expected {horizon}")
    if forecast["date"].diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("Forecast dates must be consecutive")
    if forecast["date"].min() != history["date"].max() + pd.Timedelta(days=1):
        raise ValueError("Forecast must start the day after history")
    if history["date"].isin(forecast["date"]).any():
        raise ValueError("History and forecast overlap")
    numeric_forecast = ["forecast", "lower_80", "upper_80"]
    if forecast[numeric_forecast].isna().any().any():
        raise ValueError("forecast.csv contains missing or non-numeric values")
    if history["revenue"].isna().any():
        raise ValueError("History contains missing or non-numeric values")
    if not forecast[numeric_forecast].apply(
        lambda column: pd.to_numeric(column, errors="coerce").notna().all()
    ).all():
        raise ValueError("forecast.csv contains non-numeric values")
    if not pd.api.types.is_bool_dtype(forecast["reliable"]):
        raise ValueError("forecast.csv contains a non-boolean reliable column")
    if not pd.to_numeric(history["revenue"], errors="coerce").notna().all():
        raise ValueError("History contains non-numeric values")
    if (forecast["lower_80"] > forecast["forecast"]).any() or (
        forecast["forecast"] > forecast["upper_80"]
    ).any():
        raise ValueError("Bounds must satisfy lower_80 <= forecast <= upper_80")
    closed = forecast["date"].dt.dayofweek.eq(5) | forecast["date"].isin(closure_dates)
    if forecast.loc[closed, numeric_forecast].ne(0).any().any():
        raise ValueError("Known closed days must have zero forecast and bounds")
    return forecast, history


def _validate_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    config = _load_config()
    forecast = _read_csv("forecast.csv", REQUIRED_FORECAST, ["date"])
    history_name = "daily_revenue.csv" if (DATA_DIR / "daily_revenue.csv").exists() else "history.csv"
    history = _read_csv(history_name, REQUIRED_HISTORY, ["date"])
    metrics = _read_csv("test_results.csv", REQUIRED_METRICS)
    for frame in (forecast, history):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["forecast", "lower_80", "upper_80"]:
        forecast[column] = pd.to_numeric(forecast[column], errors="coerce")
    history["revenue"] = pd.to_numeric(history["revenue"], errors="coerce")
    _validate_forecast_data(forecast, history, config)
    forecast = forecast.sort_values("date").reset_index(drop=True)
    history = history.sort_values("date").reset_index(drop=True)
    for column in REQUIRED_METRICS - {"model"}:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    if metrics[list(REQUIRED_METRICS - {"model"})].isna().any().any():
        raise ValueError("test_results.csv contains invalid metrics")
    params_path = DATA_DIR / "best_params.json"
    if not params_path.exists():
        params_path = ROOT / "configs" / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing champion parameters: {params_path}")
    with params_path.open(encoding="utf-8") as handle:
        champion = json.load(handle).get("champion")
    if not champion:
        raise ValueError("Champion is missing from exported parameters")
    return forecast, history, metrics, str(champion)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    return _validate_data()


def _gbp(value: float) -> str:
    return f"£{value:,.0f}"


def _humanize_regime(value: str) -> str:
    return REGIME_LABELS.get(value, value.replace("_", " ").capitalize())


def _forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    first_forecast = forecast["date"].min()
    plotted_history = history.loc[history["date"] < first_forecast].tail(90)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=plotted_history["date"], y=plotted_history["revenue"], mode="lines",
        name="History", line={"color": "#111827", "width": 2},
        hovertemplate="%{x|%d %b %Y}<br>£%{y:,.0f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["upper_80"], mode="lines",
        line={"width": 0}, showlegend=False, hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["lower_80"], mode="lines",
        fill="tonexty", fillcolor="rgba(37, 99, 235, 0.15)",
        line={"width": 0}, name="80% interval", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecast"], mode="lines",
        name="Forecast", line={"color": "#2563EB", "width": 2.5},
        hovertemplate="%{x|%d %b %Y}<br>£%{y:,.0f}<extra></extra>",
    ))
    for reliable, label, color in [(True, "Reliable", "#2563EB"), (False, "Low confidence", "#93C5FD")]:
        points = forecast.loc[forecast["reliable"].eq(reliable)]
        if not points.empty:
            figure.add_trace(go.Scatter(
                x=points["date"], y=points["forecast"], mode="markers", name=label,
                marker={"size": 6, "color": color},
                hovertemplate="%{x|%d %b %Y}<br>£%{y:,.0f}<extra></extra>",
            ))
    zero = forecast["forecast"].eq(0).tolist()
    start = None
    for index, is_zero in enumerate(zero + [False]):
        if is_zero and start is None:
            start = index
        elif start is not None:
            if index - start >= 3:
                figure.add_vrect(
                    x0=forecast.iloc[start]["date"], x1=forecast.iloc[index - 1]["date"],
                    fillcolor="#F3F4F6", opacity=0.8, line_width=0,
                    annotation_text="Closure", annotation_position="top left",
                )
            start = None
    figure.add_vline(x=first_forecast, line_dash="dash", line_color="#6B7280")
    figure.update_layout(
        template="plotly_white", height=430,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified", font={"family": "system-ui"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": None, "showgrid": False},
        yaxis={"title": "Revenue (£)", "tickprefix": "£", "showgrid": True,
               "gridcolor": "#E5E7EB"},
    )
    return figure


def _forecast_tab(forecast: pd.DataFrame, history: pd.DataFrame) -> None:
    st.subheader("Forecast")
    st.plotly_chart(_forecast_chart(history, forecast), width="stretch",
                    config={"displayModeBar": False})
    table = forecast[["date", "forecast", "lower_80", "upper_80", "reliable"]].copy()
    table["date"] = table["date"].dt.strftime("%d %b %Y")
    table["status"] = table.pop("reliable").map({True: "Reliable", False: "Low confidence"})
    for column in ["forecast", "lower_80", "upper_80"]:
        table[column] = table[column].map(_gbp)
    st.dataframe(table.rename(columns={
        "date": "Date", "forecast": "Forecast", "lower_80": "Lower 80%",
        "upper_80": "Upper 80%", "status": "Status",
    }), hide_index=True, width="stretch")
    st.download_button("Download forecast CSV", forecast.to_csv(index=False),
                       "forecast.csv", "text/csv", width="content")


def _performance_tab(metrics: pd.DataFrame) -> None:
    st.subheader("Test set performance")
    models = ["baseline", "xgboost", "prophet"]
    display = metrics.loc[metrics["model"].isin(models)].copy()
    display["model"] = display["model"].str.title().replace({"Xgboost": "XGBoost"})
    for bucket in ["wape", "1-7_wape", "8-14_wape", "15-28_wape"]:
        if bucket in display:
            display[bucket] = display[bucket].map(lambda value: f"{value:.1%}")
    for column in ["mae", "bias"]:
        display[column] = display[column].map(_gbp)
    columns = ["model", "wape", "mae", "bias", "1-7_wape", "8-14_wape", "15-28_wape"]
    display = display[[column for column in columns if column in display]]
    st.dataframe(display.rename(columns={
        "model": "Model", "wape": "WAPE", "mae": "MAE", "bias": "Bias",
        "1-7_wape": "Days 1–7 WAPE", "8-14_wape": "Days 8–14 WAPE",
        "15-28_wape": "Days 15–28 WAPE",
    }), hide_index=True, width="stretch")
    origins = _read_csv("test_per_origin.csv", {"model", "origin", "regime", "wape"}, ["origin"])
    figure = go.Figure()
    colors = {"baseline": "#9CA3AF", "xgboost": "#6B7280", "prophet": "#2563EB"}
    for model in models:
        points = origins.loc[origins["model"].eq(model)]
        if not points.empty:
            figure.add_trace(go.Scatter(
                x=points["origin"], y=points["wape"], mode="lines+markers",
                name=model.title().replace("Xgboost", "XGBoost"),
                line={"color": colors[model]},
                hovertemplate="%{x|%d %b %Y}<br>WAPE %{y:.1%}<extra></extra>",
            ))
    figure.update_layout(
        template="plotly_white", height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified", font={"family": "system-ui"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": None, "showgrid": False},
        yaxis={"title": "WAPE", "tickformat": ".0%", "showgrid": True,
               "gridcolor": "#E5E7EB"},
    )
    st.subheader("WAPE by test origin")
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _method_tab(forecast: pd.DataFrame, history: pd.DataFrame, champion: str) -> None:
    st.subheader("Data and method")
    st.markdown(
        "- Data: UCI Online Retail, 541,909 transaction rows (2010-12-01 to 2011-12-09), "
        "cleaned to 519,726 rows (£9.82M revenue) and aggregated to 373 daily observations."
    )
    st.markdown(
        "- Evaluation: rolling-origin backtest with a 28-day horizon and weekly origins: "
        "23 development folds for model selection and 11 untouched test folds (Sep to Nov 2011)."
    )
    st.markdown(
        "- Selection: the rule was fixed before the test run. A challenger must beat the "
        "seasonal-naive baseline on pooled development WAPE and win on at least 12 of 23 "
        f"development origins. The selected model is **{champion.title()}**."
    )
    st.markdown(
        "- Intervals: 80% bounds come from the 10th and 90th percentiles of development "
        "backtest errors per horizon bucket. Known closure days are forced to zero and days "
        "right after a closure are flagged low confidence."
    )
    st.markdown(
        "- Limitations: only 373 days of history, overlapping test windows, a negative bias "
        "on the test period, and interval coverage not verified out of sample."
    )
    eda_path = DATA_DIR / "eda_summary.csv"
    if eda_path.exists():
        eda = pd.read_csv(eda_path)
        weekday = eda.loc[eda["dimension"].eq("weekday")].copy()
        if not weekday.empty:
            st.subheader("Average revenue by weekday")
            figure = go.Figure(go.Bar(x=weekday["key"], y=weekday["revenue"], marker_color="#2563EB"))
            figure.update_layout(
                template="plotly_white", height=300, font={"family": "system-ui"},
                xaxis={"title": None, "showgrid": False},
                yaxis={"title": "Revenue (£)", "tickprefix": "£", "showgrid": True,
                       "gridcolor": "#E5E7EB"},
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(
        page_title="Retail Revenue Forecast", layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""
    <style>
    .block-container { max-width: 1100px; padding-top: 1.5rem; }
    footer { visibility: hidden; }
    div[data-testid="stMetric"] {
        border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; box-shadow: none;
    }
    </style>
    """, unsafe_allow_html=True)
    try:
        forecast, history, metrics, champion = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    prophet = metrics.loc[metrics["model"].eq("prophet")].iloc[0]
    baseline = metrics.loc[metrics["model"].eq("baseline")].iloc[0]
    trading = forecast.loc[forecast["forecast"].gt(0)]
    forecast_total = forecast["forecast"].sum()
    average_forecast = forecast_total / len(trading) if len(trading) else 0
    wape_delta = (prophet["wape"] - baseline["wape"]) * 100
    st.title("Retail Revenue Forecast")
    st.caption("28-day daily revenue forecast on the UCI Online Retail dataset")
    st.caption(f"Data through {history['date'].max():%d %b %Y}")
    cards = st.columns(4)
    cards[0].metric("28-day forecast total", _gbp(forecast_total), f"{len(trading)} trading days")
    cards[1].metric("Average forecast per trading day", _gbp(average_forecast))
    cards[2].metric(
        "Prophet test WAPE", f"{prophet['wape']:.1%}",
        f"{wape_delta:+.1f} pp vs seasonal naive", delta_color="inverse",
    )
    cards[3].metric("Reliable forecast days", f"{int(forecast['reliable'].sum())}")
    forecast_tab, performance_tab, method_tab = st.tabs(
        ["Forecast", "Model performance", "Data and method"]
    )
    with forecast_tab:
        _forecast_tab(forecast, history)
    with performance_tab:
        _performance_tab(metrics)
    with method_tab:
        _method_tab(forecast, history, champion)


if __name__ == "__main__":
    main()
