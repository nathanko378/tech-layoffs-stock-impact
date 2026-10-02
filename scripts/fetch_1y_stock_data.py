"""Legacy approximately one-year helper; requires 252 completed trading sessions."""
from fetch_5d_stock_data import changes


def price_change_1y(ticker, start_dt):
    return changes(ticker, start_dt, horizon=252)[0]


def percent_change_1y(ticker, start_dt):
    return changes(ticker, start_dt, horizon=252)[1]
