import yfinance as yf
import pandas as pd
import requests_cache
from datetime import timedelta

# Setup caching for yfinance to prevent rate limiting
session = requests_cache.CachedSession(
    'yfinance_cache',
    expire_after=timedelta(minutes=15),
    allowable_methods=['GET', 'POST']
)

def fetch_data(symbol: str, interval: str = '1d', period: str = '1y', fetch_daily: bool = False) -> pd.DataFrame:
    """
    Fetches historical market data from yfinance with caching.
    """
    ticker = yf.Ticker(symbol, session=session)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    if fetch_daily and interval != '1d':
        daily_df = ticker.history(period=period, interval='1d')
        return df, daily_df

    return df
