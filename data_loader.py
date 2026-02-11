import yfinance as yf
import pandas as pd

def fetch_data(symbol: str, interval: str = '1d', period: str = '1y', fetch_daily: bool = False) -> pd.DataFrame:
    """
    Fetches historical market data from yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD').
        interval: Data interval (e.g., '1h', '1d').
        period: Time period (e.g., '1y', 'max').
        fetch_daily: If True and interval is not '1d', also returns daily data for higher-TF analysis.

    Returns:
        DataFrame or tuple of (DataFrame, Daily DataFrame).
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    if fetch_daily and interval != '1d':
        daily_df = ticker.history(period=period, interval='1d')
        return df, daily_df

    return df
