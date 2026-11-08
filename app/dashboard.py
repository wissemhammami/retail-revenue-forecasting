"""Presentation dashboard for the precomputed retail revenue forecasts."""

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


def _read_csv(name: str, required: set[str], dates: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Export requis absent : {path}")
    frame = pd.read_csv(path, parse_dates=dates or [])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} : colonnes manquantes : {', '.join(sorted(missing))}")
    return frame


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration absente : {CONFIG_PATH}")
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
            raise ValueError(f"{label} contient des dates invalides")
        if frame["date"].duplicated().any():
            raise ValueError(f"{label} contient des dates dupliquées")
        if not frame["date"].is_monotonic_increasing:
            raise ValueError(f"{label} doit être chronologique")
    if len(forecast) != horizon:
        raise ValueError(f"Horizon inattendu : {len(forecast)} lignes, attendu {horizon}")
    if forecast["date"].diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("Les dates de prévision doivent être consécutives")
    if forecast["date"].min() != history["date"].max() + pd.Timedelta(days=1):
        raise ValueError("La prévision doit commencer le lendemain de l'historique")
    if history["date"].isin(forecast["date"]).any():
        raise ValueError("Chevauchement entre historique et prévision")
    numeric_forecast = ["forecast", "lower_80", "upper_80"]
    if forecast[numeric_forecast].isna().any().any():
        raise ValueError("forecast.csv contient des valeurs manquantes ou non numériques")
    if history["revenue"].isna().any():
        raise ValueError("L'historique contient des valeurs manquantes ou non numériques")
    if not forecast[numeric_forecast].apply(
        lambda column: pd.to_numeric(column, errors="coerce").notna().all()
    ).all():
        raise ValueError("forecast.csv contient des valeurs non numériques")
    if not pd.api.types.is_bool_dtype(forecast["reliable"]):
        raise ValueError("forecast.csv contient une colonne reliable non booléenne")
    if not pd.to_numeric(history["revenue"], errors="coerce").notna().all():
        raise ValueError("L'historique contient des valeurs non numériques")
    if (forecast["lower_80"] > forecast["forecast"]).any() or (
        forecast["forecast"] > forecast["upper_80"]
    ).any():
        raise ValueError("Les bornes doivent respecter lower_80 <= forecast <= upper_80")
    closed = forecast["date"].dt.dayofweek.eq(5) | forecast["date"].isin(closure_dates)
    if forecast.loc[closed, numeric_forecast].ne(0).any().any():
        raise ValueError("Les jours fermés connus doivent avoir forecast et bornes à zéro")
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
    forecast = forecast.loc[forecast["reliable"]].reset_index(drop=True)
    history = history.sort_values("date").reset_index(drop=True)
    for column in REQUIRED_METRICS - {"model"}:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    if metrics[list(REQUIRED_METRICS - {"model"})].isna().any().any():
        raise ValueError("test_results.csv contient des métriques invalides")

    params_path = DATA_DIR / "best_params.json"
    if not params_path.exists():
        params_path = ROOT / "configs" / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Paramètres du champion absents : {params_path}")
    with params_path.open(encoding="utf-8") as handle:
        champion = json.load(handle).get("champion")
    if not champion:
        raise ValueError("Le champion est absent des paramètres exportés")
    return forecast, history, metrics, str(champion)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    return _validate_data()


def _gbp(value: float) -> str:
    return f"£{value:,.0f}"


def _forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    first_forecast = forecast["date"].min()
    plotted_history = history.loc[history["date"] < first_forecast]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=plotted_history["date"], y=plotted_history["revenue"],
        mode="lines", name="Historique", line={"color": "#2F6B9A", "width": 2},
        hovertemplate="%{x|%d %b %Y}<br>£%{y:,.0f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["upper_80"], mode="lines",
        line={"width": 0}, showlegend=False, hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["lower_80"], mode="lines",
        fill="tonexty", fillcolor="rgba(67, 160, 160, 0.18)",
        line={"width": 0}, name="Intervalle 80 %", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecast"], mode="lines+markers",
        name="Prévision", line={"color": "#159A9C", "width": 2.5},
        marker={"size": 4},
        hovertemplate="%{x|%d %b %Y}<br>£%{y:,.0f}<extra></extra>",
    ))
    figure.add_vline(x=first_forecast, line_dash="dash", line_color="#1F3A5F")
    figure.update_layout(
        height=430, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": None, "showgrid": False},
        yaxis={"title": "Revenu (£)", "tickprefix": "£", "showgrid": True,
               "gridcolor": "#E8EDF2"},
    )
    return figure


def _overview(forecast: pd.DataFrame, history: pd.DataFrame, metrics: pd.DataFrame, champion: str) -> None:
    st.header("Executive Overview")
    if "is_trading_day" in history:
        trading_mask = history["is_trading_day"].fillna(False).astype(bool)
    else:
        trading_mask = pd.Series(True, index=history.index)
    trading = history.loc[trading_mask]
    best_day = trading.loc[trading["revenue"].idxmax()]
    cards = st.columns(5)
    values = [
        ("Total historique", history["revenue"].sum()),
        ("Moyenne journalière", history["revenue"].mean()),
        ("Prévision 7 jours", forecast.head(7)["forecast"].sum()),
        ("Prévision fiable", forecast["forecast"].sum()),
    ]
    for card, (label, value) in zip(cards[:4], values):
        card.metric(label, _gbp(value))
    cards[4].metric("Meilleur jour ouvré", f"{best_day['date']:%d %b %Y}")
    cards[4].caption(f"Revenu : {_gbp(best_day['revenue'])}")
    left, right = st.columns([2.4, 1])
    with left:
        st.plotly_chart(_forecast_chart(history, forecast), width="stretch")
    with right:
        st.subheader("Résumé de prévision")
        st.metric("Total prévision fiable", _gbp(forecast["forecast"].sum()))
        st.caption(
            f"Intervalle agrégé approximatif : {_gbp(forecast['lower_80'].sum())} – "
            f"{_gbp(forecast['upper_80'].sum())}"
        )
        st.write(f"Modèle retenu : **{champion.title()}**")
        st.caption("Les intervalles représentent une incertitude empirique, pas une garantie.")


def _recent_forecast(forecast: pd.DataFrame) -> None:
    st.header("Recent Forecast")
    table = forecast.head(7).drop(columns=["reliable"], errors="ignore").copy()
    table.insert(1, "Day", table["date"].dt.day_name())
    table["date"] = table["date"].dt.strftime("%d %b %Y")
    for column in ["forecast", "lower_80", "upper_80"]:
        table[column] = table[column].round().map(_gbp)
    st.dataframe(table.rename(columns={
        "date": "Date", "forecast": "Forecast", "lower_80": "Lower 80%", "upper_80": "Upper 80%"
    }), hide_index=True, width="stretch")
    st.download_button(
        "Télécharger les jours fiables",
        forecast.to_csv(index=False),
        "forecast.csv",
        "text/csv",
    )


def _performance(metrics: pd.DataFrame) -> None:
    st.header("Model Performance")
    st.caption("Métriques issues de la période de test tenue à l'écart du réglage ; horizon agrégé de 28 jours.")
    display = metrics.loc[metrics["model"].isin(["baseline", "prophet", "xgboost"]),
                          ["model", "mae", "rmse", "wape", "bias"]].copy()
    display["wape"] = display["wape"].map(lambda value: f"{value:.1%}")
    display["mae"] = display["mae"].map(_gbp)
    display["rmse"] = display["rmse"].map(_gbp)
    display["bias"] = display["bias"].map(_gbp)
    st.dataframe(display.rename(columns={
        "model": "Model", "mae": "MAE", "rmse": "RMSE", "wape": "WAPE", "bias": "Bias (forecast - actual)"
    }), hide_index=True, width="stretch")
    st.caption("MAE, RMSE et WAPE : plus bas est meilleur. Biais positif = sur-prévision ; négatif = sous-prévision.")


def _insights(forecast: pd.DataFrame, history: pd.DataFrame) -> None:
    st.header("Quick Insights")
    insight_columns = st.columns(3)
    weekday = history.assign(weekday=history["date"].dt.day_name()).groupby("weekday")["revenue"].mean()
    strongest = weekday.idxmax()
    insight_columns[0].write(f"**Jour le plus fort**  \n{strongest}, avec {_gbp(weekday.max())} en moyenne.")
    insight_columns[1].write(f"**Prévision à 7 jours**  \n{_gbp(forecast.head(7)['forecast'].sum())}.")
    trend = history.tail(28)["revenue"].mean() - history.tail(56).head(28)["revenue"].mean()
    direction = "en hausse" if trend > 0 else "en baisse"
    insight_columns[2].write(f"**Dernière tendance disponible**  \nMoyenne des 28 derniers jours {direction}.")


def main() -> None:
    st.set_page_config(page_title="Retail Revenue Forecasting", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #FFFFFF; color: #172A46; }
    [data-testid="stSidebar"] { background: #F4F7FA; width: 220px; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] { border: 1px solid #DCE5EC; border-radius: 8px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)
    try:
        forecast, history, metrics, champion = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.title("Retail Revenue Forecasting")
    st.subheader("Historical performance, model evaluation and revenue prediction")
    st.caption("Days after reopening are not forecast because the post-closure level is not validated.")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Recent Forecast", "Model Performance", "Quick Insights"])
    st.sidebar.caption(f"Champion model : {champion.title()}")
    if page == "Overview":
        _overview(forecast, history, metrics, champion)
    elif page == "Recent Forecast":
        _recent_forecast(forecast)
    elif page == "Model Performance":
        _performance(metrics)
    else:
        _insights(forecast, history)


if __name__ == "__main__":
    main()
