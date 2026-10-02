"""Legacy one-year analysis is superseded by the trading-session ML pipeline."""
if __name__ == "__main__":
    raise SystemExit("Run python -m src.pipeline for the five-session predictor. Use --horizon 252 for an approximately one-year trading-session model with separate --output artifacts_1y.")
