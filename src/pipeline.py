"""Run with python -m src.pipeline [--refresh] [--offline]."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, load_events, load_prices, build_dataset
from .monitor import drift_report
from .model import chronological_split, fit_models, metrics, explain, predict


def run(args):
    input_events = len(pd.read_csv(args.events))
    events = load_events(args.events)
    if events.empty:
        raise ValueError("No valid equity events remain after cleaning. Review your input and ticker mappings.")
    cache = ROOT / "data/prices"
    start = (events.event_date.min() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    as_of = pd.Timestamp(args.as_of).normalize()
    benchmark = load_prices(args.benchmark, start, args.as_of, cache, args.refresh)
    prices, failures = {}, {}
    for ticker in sorted(events.Ticker.unique()):
        try:
            if args.offline and not (cache / f"{ticker}.csv").exists():
                raise ValueError("Missing offline price cache")
            prices[ticker] = load_prices(ticker, start, args.as_of, cache, args.refresh)
        except Exception as exc:
            failures[ticker] = str(exc)
        print(f"Prices: {ticker}", flush=True)
    print(f"Downloaded {len(prices)} tickers; {len(failures)} failed. Building dataset…", flush=True)
    data = build_dataset(events, prices, benchmark, args.horizon, as_of)
    labeled = data.dropna(subset=["target", "target_end"])
    train, val, test = chronological_split(labeled)
    models, winner, thresholds, validation = fit_models(train, val)
    test = test.copy()
    evaluation = {}
    for name, model in models.items():
        p = predict(model, test)
        test[f"probability_{name}"] = p
        evaluation[name] = metrics(test.target, p, thresholds[name])
    test["probability"] = test[f"probability_{winner}"]
    # Date-cluster bootstrap reflects same-day dependence; descriptive, small-sample interval.
    rng = np.random.default_rng(42)
    dates = test.prediction_date.unique()
    aucs = []
    for _ in range(300):
        sample = pd.concat([test[test.prediction_date == date] for date in rng.choice(dates, len(dates))])
        if sample.target.nunique() == 2:
            aucs.append(metrics(sample.target, sample.probability, thresholds[winner])["roc_auc"])
    explanations = explain(models[winner], train, test)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[winner], output / "model.joblib")
    joblib.dump(models, output / "models.joblib")
    train[FEATURES].to_csv(output / "background.csv", index=False)
    data.to_csv(output / "dataset.csv", index=False)
    drift_report(train, test).to_csv(output / "drift.csv", index=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    np.savez_compressed(output / "shap_values.npz", values=explanations.values,
                        base_values=explanations.base_values, data=test[FEATURES].to_numpy())
    metadata = dict(model=winner, threshold=thresholds[winner], thresholds=thresholds,
                    features=FEATURES, benchmark=args.benchmark, horizon=args.horizon,
                    as_of=args.as_of, created_at=pd.Timestamp.now(tz="UTC").isoformat(),
                    dataset_sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
                    price_coverage={ticker: {"start": str(frame.index.min().date()), "end": str(frame.index.max().date())} for ticker, frame in {**prices, args.benchmark: benchmark}.items()},
                    data_sha256=hashlib.sha256(Path(args.events).read_bytes()).hexdigest(),
                    target="Stock adjusted-close return exceeds benchmark over horizon trading sessions",
                    timing="Before first common stock/benchmark session strictly after Date Added; features end at prior close",
                    date_caveat="Date Added is a tracker-entry proxy, not a verified announcement timestamp",
                    failures=failures, input_events=input_events, excluded_events=input_events-len(events), events=len(events), usable=len(data), labeled=len(labeled),
                    validation=validation, test=evaluation,
                    roc_auc_interval=np.quantile(aucs, [.025, .975]).tolist() if aucs else None,
                    splits={name:dict(n=len(frame), start=str(frame.prediction_date.min().date()),
                                     end=str(frame.prediction_date.max().date()))
                            for name, frame in [("train", train), ("validation", val), ("test", test)]},
                    versions={name:importlib.metadata.version(name) for name in
                              ["scikit-learn", "xgboost", "shap", "pandas", "numpy"]})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    history_path = output / "run_history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    entry = dict(created_at=metadata["created_at"], model=winner, test_n=len(test),
                 roc_auc=evaluation[winner]["roc_auc"], average_precision=evaluation[winner]["average_precision"],
                 failed_tickers=len(failures), labeled_events=len(labeled))
    pd.concat([history, pd.DataFrame([entry])], ignore_index=True).to_csv(history_path, index=False)
    print(f"Saved {winner} artifacts to {output}; held-out AUC={evaluation[winner]['roc_auc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(ROOT / "data/raw/clean_layoff_data.csv"))
    parser.add_argument("--output", default=str(ROOT / "artifacts"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--as-of", default=str(pd.Timestamp.now(tz="UTC").date()))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1 or (args.offline and args.refresh):
        parser.error("Horizon must be positive; offline and refresh cannot be combined.")
    if args.offline and not (ROOT / "data/prices" / f"{args.benchmark}.csv").exists():
        parser.error("Missing offline benchmark cache.")
    run(args)

if __name__ == "__main__":
    main()
