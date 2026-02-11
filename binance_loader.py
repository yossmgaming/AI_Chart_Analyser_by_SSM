from binance.client import Client
import pandas as pd
from datetime import datetime
import requests

class BinanceLoader:
    def __init__(self, tld='com'):
        # Try different TLDs if restricted
        self.tld = tld
        try:
            self.client = Client(None, None, tld=self.tld)
        except:
            self.client = None

    def fetch_data(self, symbol: str, interval: str = '1m', limit: int = 500) -> pd.DataFrame:
        """
        Fetches historical klines from Binance.
        """
        if self.client is None:
             # Fallback to direct requests or just raise
             return self._fetch_via_requests(symbol, interval, limit)

        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }
        binance_interval = interval_map.get(interval, interval)

        try:
            klines = self.client.get_klines(symbol=symbol, interval=binance_interval, limit=limit)
        except Exception as e:
            print(f"Binance API error: {e}")
            return self._fetch_via_requests(symbol, interval, limit)

        return self._process_klines(klines)

    def _fetch_via_requests(self, symbol, interval, limit):
        # Try US endpoint if COM is blocked
        base_url = "https://api.binance.us/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        resp = requests.get(base_url, params=params)
        if resp.status_code == 200:
            return self._process_klines(resp.json())
        else:
            # If both fail, we might be totally blocked or symbol is wrong
            raise ValueError(f"Could not fetch Binance data for {symbol}. Service might be restricted or symbol is invalid.")

    def _process_klines(self, klines):
        df = pd.DataFrame(klines, columns=[
            'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ])

        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df.set_index('Time', inplace=True)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

if __name__ == "__main__":
    loader = BinanceLoader()
    try:
        df = loader.fetch_data("BTCUSDT", interval='1h', limit=10)
        print(df.tail())
    except Exception as e:
        print(e)
