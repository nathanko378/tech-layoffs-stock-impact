"""Clean source announcements using reviewed ticker mappings; never guess a ticker."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "data/raw/Layoffs.fyi  - Tech Layoffs Tracker.csv"))
    parser.add_argument("--mapping", default=str(ROOT / "data/raw/ticker_mapping.csv"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/events.csv"))
    args = parser.parse_args()
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        existing = pd.read_csv(ROOT / "data/raw/clean_layoff_data.csv")
        existing[["Company", "Ticker"]].drop_duplicates().to_csv(mapping_path, index=False)
        raise SystemExit(f"Created {mapping_path}. Review its company/ticker matches, then rerun this command.")
    mapping = pd.read_csv(mapping_path)[["Company", "Ticker"]].drop_duplicates()
    if mapping.Company.duplicated().any():
        raise ValueError("Each company must have one reviewed ticker mapping.")
    raw = pd.read_csv(args.source)
    raw = raw.loc[raw.Stage == "Post-IPO"].copy()
    # Prefer announcement date where available, falling back explicitly to tracker-entry date.
    raw["Date Added"] = pd.to_datetime(raw["Date"], errors="coerce").fillna(pd.to_datetime(raw["Date Added"], errors="coerce"))
    merged = raw.merge(mapping, on="Company", how="left", validate="many_to_one")
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    merged[merged.Ticker.isna()].to_csv(destination.with_name("unmapped_events.csv"), index=False)
    merged.dropna(subset=["Ticker"]).to_csv(destination, index=False)
    print(f"Saved {merged.Ticker.notna().sum()} mapped events; {merged.Ticker.isna().sum()} require ticker review.")

if __name__ == "__main__":
    main()
