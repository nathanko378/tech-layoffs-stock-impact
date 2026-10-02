"""Predict a new event from cached or downloaded pre-event market data."""
import argparse
import json
import re
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, load_events, load_prices, market_features
from .model import predict


def predict_event(ticker, event_date, layoff_count, layoff_percent, artifacts=ROOT / "artifacts"):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}(?:[.\-][AB])?", ticker):
        raise ValueError("Enter a valid ticker symbol.")
    date = pd.Timestamp(event_date).normalize()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if date > today:
        raise ValueError("Future events cannot have observed pre-event market features.")
    if any(v is not None and not np.isfinite(v) for v in [layoff_count, layoff_percent]):
        raise ValueError("Layoff inputs must be finite or unknown.")
    if layoff_count is not None and layoff_count < 0:
        raise ValueError("Layoff count must be nonnegative.")
    if layoff_percent is not None and not 0 <= layoff_percent <= 100:
        raise ValueError("Layoff percentage must be between 0 and 100.")
    meta = json.loads((artifacts / "metadata.json").read_text())
    # Match training: use completed closes through the date-only event date.
    start = (date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    cutoff = min(date + pd.Timedelta(days=1), today)
    end = cutoff.strftime("%Y-%m-%d")
    cache = ROOT / "data/inference_prices"
    stock = load_prices(ticker, start, end, cache, refresh=True)
    benchmark = load_prices(meta["benchmark"], start, end, cache, refresh=True)
    if len(stock) < 61 or len(benchmark) < 21:
        raise ValueError("Insufficient pre-event market history.")
    history = pd.read_csv(artifacts / "dataset.csv", parse_dates=["event_date"])
    history = history[(history.Ticker == ticker) & (history.event_date < date)]
    row = dict(layoff_count=layoff_count, layoff_percent=layoff_percent,
               count_missing=int(layoff_count is None), percent_missing=int(layoff_percent is None),
               prior_events=len(history), days_since_event=(date-history.event_date.max()).days if len(history) else None)
    row.update(market_features(stock, cutoff))
    row.update(market_features(benchmark, cutoff, "market_"))
    frame = pd.DataFrame([row], columns=FEATURES).astype(float)
    probability = float(predict(joblib.load(artifacts / "model.joblib"), frame)[0])
    return probability, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("event_date")
    parser.add_argument("--layoff-count", type=float)
    parser.add_argument("--layoff-percent", type=float)
    args = parser.parse_args()
    probability, frame = predict_event(args.ticker, args.event_date, args.layoff_count, args.layoff_percent)
    print(json.dumps({"probability": probability, "features": frame.iloc[0].where(frame.iloc[0].notna(), None).to_dict()}, indent=2))

if __name__ == "__main__":
    main()
