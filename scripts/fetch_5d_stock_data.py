"""Legacy helper with explicit trading-session timing and adjusted prices."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load_prices


def changes(ticker, start_dt, horizon=5):
    date = pd.Timestamp(start_dt)
    prices = load_prices(ticker, (date-pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
                         (date+pd.Timedelta(days=max(30, horizon*2))).strftime("%Y-%m-%d"),
                         ROOT / "data/legacy_prices", refresh=True)
    index = prices.index.searchsorted(date, side="right")
    if index == 0 or index+horizon-1 >= len(prices):
        raise ValueError("Full return horizon unavailable.")
    before, after = prices.Close.iloc[index-1], prices.Close.iloc[index+horizon-1]
    return float(after-before), float((after/before-1)*100)


def price_change_5d(ticker, start_dt):
    return changes(ticker, start_dt)[0]


def percent_change_5d(ticker, start_dt):
    return changes(ticker, start_dt)[1]
