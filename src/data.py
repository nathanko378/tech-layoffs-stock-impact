from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["layoff_count", "layoff_percent", "count_missing", "percent_missing",
            "prior_events", "days_since_event", "return_5", "return_20", "return_60",
            "volatility_20", "relative_volume", "market_return_20", "market_volatility_20"]


def load_events(path):
    df = pd.read_csv(path).rename(columns={"%": "% Laid Off"})
    df["event_date"] = pd.to_datetime(df["Date Added"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    for source, target in [("# Laid Off", "layoff_count"), ("% Laid Off", "layoff_percent")]:
        df[target] = pd.to_numeric(df[source].astype(str).str.replace("%", "", regex=False)
                                   .str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["event_date"])
    df = df[df.Ticker.str.match(r"^[A-Z]{1,5}(?:[.\-][AB])?$") & ~df.Ticker.isin(["N/A", "NAN"])]
    df = df[(df.layoff_count.isna() | (df.layoff_count >= 0)) &
            (df.layoff_percent.isna() | df.layoff_percent.between(0, 100))]
    df = df.drop_duplicates(["Ticker", "event_date"]).sort_values(["event_date", "Ticker"]).reset_index(drop=True)
    df["event_id"] = df.Ticker + ":" + df.event_date.dt.strftime("%Y-%m-%d")
    df["count_missing"] = df.layoff_count.isna().astype(int)
    df["percent_missing"] = df.layoff_percent.isna().astype(int)
    df["prior_events"] = df.groupby("Ticker").cumcount()
    df["days_since_event"] = df.groupby("Ticker").event_date.diff().dt.days
    return df


def load_prices(ticker, start, end, cache_dir, refresh=False):
    """Cache adjusted daily prices; end is exclusive. Offline runs reuse the cache."""
    import yfinance as yf
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    path = cache_dir / f"{key}.csv"
    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        if prices.empty:
            raise ValueError(f"No prices returned for {ticker}")
        prices = prices[["Close", "Volume"]].copy()
        prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
        prices.index.name = "Date"
        prices.to_csv(path)
    prices = prices.sort_index().loc[lambda x: ~x.index.duplicated()]
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end))]
    prices = prices.dropna(subset=["Close"])
    prices = prices.loc[prices.Close > 0]
    if prices.empty:
        raise ValueError(f"No valid cached prices in requested interval for {ticker}; rerun with --refresh")
    return prices


def market_features(prices, date, prefix=""):
    history = prices.loc[prices.index < date]
    result = {}
    for n in ([20] if prefix else [5, 20, 60]):
        result[f"{prefix}return_{n}"] = (history.Close.iloc[-1] / history.Close.iloc[-n-1] - 1
                                           if len(history) > n else np.nan)
    result[f"{prefix}volatility_20"] = (history.Close.pct_change().tail(20).std()
                                         if len(history) >= 21 else np.nan)
    if not prefix:
        result["relative_volume"] = (history.Volume.iloc[-1] / history.Volume.tail(20).mean()
                                      if len(history) >= 20 and history.Volume.tail(20).mean() > 0 else np.nan)
    return result


def build_dataset(events, prices, benchmark, horizon=5, as_of=None):
    """Predict before first session strictly after event date; five close-to-close sessions.

    The reference close is the session before prediction. Future prices are labels only.
    Date Added is a proxy for announcement availability, explicitly recorded in metadata.
    """
    as_of = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC").date()).normalize()
    rows = []
    for _, event in events.iterrows():
        stock = prices.get(event.Ticker)
        if stock is None:
            continue
        stock = stock.loc[stock.index < as_of]
        market = benchmark.loc[benchmark.index < as_of]
        sessions = stock.index.intersection(market.index).sort_values()
        pos = sessions.searchsorted(event.event_date, side="right")
        if pos == 0 or pos >= len(sessions):
            continue
        prediction_date, reference = sessions[pos], sessions[pos-1]
        row = event.to_dict()
        row.update(market_features(stock, prediction_date))
        row.update(market_features(market, prediction_date, "market_"))
        row.update(prediction_date=prediction_date, target_end=pd.NaT,
                   stock_return=np.nan, benchmark_return=np.nan, excess_return=np.nan, target=np.nan)
        end_pos = pos + horizon - 1
        if end_pos < len(sessions):
            end_date = sessions[end_pos]
            sr = stock.loc[end_date, "Close"] / stock.loc[reference, "Close"] - 1
            br = market.loc[end_date, "Close"] / market.loc[reference, "Close"] - 1
            row.update(target_end=end_date, stock_return=sr, benchmark_return=br,
                       excess_return=sr-br, target=int(sr > br))
        rows.append(row)
    if not rows:
        raise ValueError("No usable events. Check ticker mappings and cached price coverage.")
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
