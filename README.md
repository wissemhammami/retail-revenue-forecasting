# Retail Revenue Forecasting

[![CI](https://github.com/wissemhammami/retail-revenue-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/wissemhammami/retail-revenue-forecasting/actions/workflows/ci.yml)

28-day daily revenue forecasting on the UCI Online Retail dataset. A seasonal-naive baseline, Prophet and XGBoost are compared with rolling-origin backtesting; the champion is served as a Streamlit dashboard.

## Live demo

- Live dashboard: https://retail-wissem.streamlit.app/

The first load after a period of inactivity can take about 30 seconds.

## Problem

Forecast a UK online retailer's daily revenue 28 days ahead, including known year-end closures, to support stock and staffing planning. Data: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), 541,909 transaction rows (2010-12-01 to 2011-12-09), cleaned to 519,726 rows (£9.82M revenue) and aggregated to 373 daily observations.

## Results

Rolling-origin backtest, 28-day horizon, weekly origins: 23 development folds for model selection, 11 untouched test folds (Sep to Nov 2011).

| Model | Dev WAPE | Test WAPE | Test MAE | Test bias |
|---|---:|---:|---:|---:|
| Seasonal naive | 36.2% | 35.0% | £13,988 | −£4,715 |
| XGBoost | 33.2% | 29.0% | £11,599 | −£5,207 |
| **Prophet (champion)** | **30.0%** | **25.3%** | **£10,103** | −£4,825 |

Selection rule, fixed before the test run: a challenger must have a lower pooled WAPE than the baseline and win on at least 12 of 23 development origins; among qualified challengers the lowest WAPE wins. Prophet beats the baseline on all 11 test origins. The test script refuses to overwrite an existing result file.

## Pipeline

1. **Clean:** drop duplicates, cancellations, invalid prices and quantities, and non-product codes.
2. **Aggregate:** daily revenue on a full calendar, with configured closures and non-trading days.
3. **Features:** calendar flags and lags (7, 14, 28) with shifted rolling means, built without future revenue.
4. **Evaluate:** rolling-origin folds, WAPE, MAE, RMSE, bias, per-horizon and per-origin breakdowns.
5. **Forecast:** champion fitted on the full history; 80% intervals from the 10th and 90th development residual quantiles per horizon bucket; closure days forced to zero; days right after a closure flagged unreliable.
6. **Serve:** Streamlit dashboard (`app/`), Docker image, CI.

## Project structure

```text
app/           Streamlit dashboard + its own requirements.txt
configs/       config.yaml, best_params.json
dashboard_data/ small CSV/JSON files read by the dashboard
data/          raw/ and interim/ are gitignored; processed/ is committed (small)
notebooks/     EDA notebook
scripts/       pipeline entry points
src/           retail_forecast package
tests/         pytest suite
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app/dashboard.py
```

Docker (dashboard only):

```bash
docker build -t retail-forecast .
docker run -p 8501:8501 retail-forecast
```

CI runs the tests on Python 3.12, then builds the image and checks the Streamlit health endpoint on every push.

## Reproduce the results

The raw and cleaned transaction files are gitignored (too large). Run in this order:

```bash
make data        # download and convert the UCI dataset
make clean
make series
make features
make cv          # development CV and champion selection
make test-run    # one-shot test evaluation
make forecast
```

## Deployment

Streamlit Community Cloud, entrypoint `app/dashboard.py`, Python 3.12. Dependencies come from `app/requirements.txt`.

## Limitations

- Only 373 days of history: yearly seasonality is not modelled, so the December forecast extends the weekly pattern and known closures.
- The 11 test origins use overlapping 28-day windows, so the gain over the baseline is less certain than the headline numbers suggest.
- All models under-forecast on the test period (bias of about −£4.8k per day).
- Intervals are calibrated on development residuals; out-of-sample coverage has not been verified yet.
- Public dataset from a single retailer; no external regressors (promotions, prices).

## Author

**Wissem Hammami**  
Machine Learning Engineer | Data Science | ESSAI, University of Carthage  
[LinkedIn](https://www.linkedin.com/in/wissemhammami/)