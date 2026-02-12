import yfinance as yf
import pandas as pd
from binance_loader import BinanceLoader
from deriv_loader import fetch_deriv_data

def fetch_data(symbol: str, interval: str = '1d', period: str = '1y', source: str = 'yfinance', limit: int = 500, fetch_daily: bool = False) -> pd.DataFrame:
    """
    Unified data fetcher for multiple sources.
    Sources: 'yfinance', 'binance', 'deriv'
    """
    if source == 'yfinance':
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for {symbol} on yfinance")

        if fetch_daily and interval != '1d':
            daily_df = ticker.history(period=period, interval='1d')
            return df, daily_df
        return df

    elif source == 'binance':
        loader = BinanceLoader()
        # Period mapping to limit if needed, or just use limit
        return loader.fetch_data(symbol, interval=interval, limit=limit)

    elif source == 'deriv':
        return fetch_deriv_data(symbol, interval=interval, count=limit)

    else:
        raise ValueError(f"Unsupported source: {source}")

def get_available_sources():
    return ['yfinance', 'binance', 'deriv', 'coinbase_l2', 'simulation_demo']
