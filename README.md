# Stock Impact Predictor

An event-based machine learning pipeline predicting whether a stock will outperform SPY over five trading sessions following a layoff announcement. Includes scikit-learn preprocessing, baseline/logistic regression/XGBoost comparison, chronological held-out evaluation, and a Streamlit dashboard with probability predictions and SHAP explanations.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --as-of 2026-10-02
python -m streamlit run app.py
```

If you already have the project's `.venv`, launch with `bash run_dashboard.sh` or directly with `.venv/bin/python -m streamlit run app.py`. Both guarantee Streamlit uses the project environment even when a global `streamlit` command is installed. Stop any previous dashboard process before relaunching, or use `bash run_dashboard.sh --server.port 8502` and open http://localhost:8502. In your IDE, select `.venv/bin/python` as the Python interpreter. If dependencies are missing, run `.venv/bin/python -m pip install -r requirements.txt`.

On macOS, XGBoost also requires `brew install libomp`.

Training needs internet access to Yahoo Finance on the first run. Downloads are cached in `data/prices/`. Invalid tickers and download failures are recorded in model metadata and displayed in the dashboard. The default input is the existing `data/raw/clean_layoff_data.csv`; company-to-ticker matches in this historical file have not been independently verified. Review mappings and missing coverage before interpreting scores. Legacy stock-return CSVs are not used for training.

```bash
# Reuse local price caches without internet
python -m src.pipeline --offline --as-of 2026-10-02
# Refresh prices when new outcome labels become available
python -m src.pipeline --refresh
# Predict a date-only announcement using pre-prediction prices
python -m src.predict AAPL 2026-10-01 --layoff-count 500 --layoff-percent 2
```

Use `--events PATH`, `--benchmark SYMBOL`, `--horizon N`, or `--output PATH` to change training configuration. The dashboard reads `artifacts/`. A different horizon should use a separate output directory. Five sessions is the default; 252 is approximately one year. Recent events without full outcomes remain unlabeled and are excluded from evaluation.

## Refreshing announcement data

```bash
python scripts/clean_layoff_data.py
# First run creates data/raw/ticker_mapping.csv and stops for manual review.
# Review company-to-symbol pairs, then rerun:
python scripts/clean_layoff_data.py
python -m src.pipeline --events data/processed/events.csv --refresh
```

The cleaner retains Post-IPO events, prefers the source `Date` over tracker `Date Added`, and writes unmatched companies to `data/processed/unmapped_events.csv`. The default model admits US-style equity symbols (letters and optional A/B share classes), excluding obvious cryptocurrency, futures, and foreign exchange suffixes so local-currency returns are not compared with USD SPY. This syntactic filter does not prove a company match or US listing. Missing layoff size is retained and imputed during model training. Ticker mappings must be reviewed, including exchange, listing history, and symbol changes. No automated first-search-result matching is used.

## Timing and outcome definition

Each row is a company/event-date pair. The prediction occurs before the first common stock/benchmark session strictly after the event date. Input prices end at the prior close; the target spans the following N common trading sessions, using adjusted close-to-close returns. Positive class means the stock return exceeds the contemporaneous benchmark return. Returns/volatility are decimal fractions; workforce percentage is in percentage points.

The historical default file uses tracker `Date Added` as a proxy for announcement availability. Exact announcement timestamps are unavailable. This is a documented date-only forecasting convention, not a claim of predicting the instantaneous announcement reaction or proving that layoffs caused returns. Inference for today's event excludes today's unfinished session.

Features include layoff count/percentage and missing indicators, previous layoff count and elapsed days, trailing 5/20/60-session stock returns, 20-session volatility and relative volume, and trailing benchmark return/volatility. Event-history features reflect only events available in the loaded dataset. Post-event returns and ticker identity are excluded from predictors.

## Modeling and evaluation

Distinct prediction dates are divided approximately 60%/20%/20% into training, validation, and test periods. Events on the same prediction date stay together. Training and validation rows whose outcomes extend into the next period are purged. Each period must contain at least five events and both classes; training fails clearly otherwise.

Training-only median imputation is saved inside scikit-learn pipelines. Candidates are a constant-probability baseline, regularized logistic regression, and a shallow regularized XGBoost classifier. Model selection uses validation average precision; thresholds maximize validation F1. The baseline can win. Fixed small-data hyperparameters are used rather than extensive tuning. The selected model is not refit on the test period.

The dashboard reports held-out ROC-AUC, average precision, precision, recall, F1, Brier score, prevalence, ROC/PR curves, calibration, and confusion matrix. A threshold slider is exploratory and does not change the saved validation-selected threshold. ROC-AUC intervals use an approximate date-cluster bootstrap; repeated companies, sample size, and model selection limit their interpretation. There is no guarantee that ML improves on the baseline.

SHAP uses a permutation explainer over the complete probability-producing pipeline, with training-only background observations. This works for XGBoost, logistic regression, and the baseline and explains imputations consistently. Saved test explanations power global mean absolute importance, beeswarm, and local waterfall plots. Contributions are in probability units, not log odds, and are not causal feature effects. Predicted probabilities are not automatically calibrated.

## Dashboard

- **Start here:** a plain-language introduction, an example of outperformance, a three-step guide, and a snapshot of model reliability.
- **Predict:** fetch pre-event market data for a new announcement, or edit a prepared event into a hypothetical scenario.
- **Explain:** held-out SHAP summaries and individual predictions.
- **Performance:** baseline comparison, metrics, threshold exploration, and calibration.
- **Historical predictions:** ticker filters and downloadable held-out predictions.
- **Methodology:** timing assumptions, evaluation periods, data coverage, monitoring, and expandable technical model details.

The app displays setup instructions until real training artifacts exist. Prepared training-period predictions are labeled as inference demonstrations, not performance estimates.

## Structure and artifacts

```text
src/data.py          event cleaning, price caching, features, targets
src/model.py         splits, pipelines, metrics, explanations, shared inference
src/pipeline.py      ingestion through saved evaluation artifacts
src/predict.py       new-event inference and CLI
scripts/             reviewed ticker mapping and legacy compatibility entry points
app.py               Streamlit UI
artifacts/           local model, metrics, data, predictions, SHAP values
```

Artifacts include `model.joblib`, all candidate models, `metadata.json`, `dataset.csv`, `test_predictions.csv`, training background features, and `shap_values.npz`. Metadata records feature schema, thresholds, split dates, dependency versions, failures, source hash, and training time. Artifacts and price caches are ignored by Git. Load only locally generated/trusted joblib artifacts.

## Verification

```bash
pip install pytest
MPLCONFIGDIR=/tmp/stock-impact-mpl python -m pytest -q
```

Tests cover feature isolation from future prices, five-session labels, immature outcomes, date-based purging, candidate inference, and SHAP additivity. Real evaluation requires verified company mappings and market-price downloads.

For ongoing operation, schedule `python -m src.pipeline --refresh` externally as new announcements and labels arrive. Feature missingness and population stability index are saved in `drift.csv`; `run_history.csv` records local training runs. These are descriptive checks, not automatic deployment gates. Automated promotion and an experiment registry are future extensions; retraining overwrites local artifacts and does not deploy a model to an external service.
