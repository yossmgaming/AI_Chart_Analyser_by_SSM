import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

class DerivLoader:
    def __init__(self, app_id=1089):
        self.app_id = app_id
        self.url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"

    async def _fetch_history(self, symbol, style, granularity, count=1000):
        async with websockets.connect(self.url) as websocket:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": 1,
                "style": style
            }
            if style == 'candles':
                request['granularity'] = granularity

            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            data = json.loads(response)

            if 'error' in data:
                raise ValueError(f"Deriv Error: {data['error']['message']}")

            return data

    def fetch_data(self, symbol: str, interval: str = '1m', count: int = 1000):
        """
        Fetches data from Deriv.
        Intervals supported: 1t (ticks), 1m, 5m, 1h, 1d.
        """
        loop = asyncio.get_event_loop()

        if interval == '1t':
            data = loop.run_until_complete(self._fetch_history(symbol, 'ticks', 0, count))
            history = data.get('history', {})
            df = pd.DataFrame({
                'Close': history.get('prices', []),
                'Time': history.get('times', [])
            })
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.set_index('Time', inplace=True)
            # For ticks, we mock Open, High, Low as Close
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0
            return df
        else:
            # Map intervals to granularity (seconds)
            mapping = {
                '1m': 60, '2m': 120, '3m': 180, '5m': 300,
                '10m': 600, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '8h': 28800, '1d': 86400
            }
            granularity = mapping.get(interval, 60)
            data = loop.run_until_complete(self._fetch_history(symbol, 'candles', granularity, count))
            candles = data.get('candles', [])
            df = pd.DataFrame(candles)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df.rename(columns={
                'epoch': 'Time',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close'
            }, inplace=True)
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.set_index('Time', inplace=True)
            df['Volume'] = 0 # Deriv doesn't provide volume for synthetics usually
            return df

def fetch_deriv_data(symbol: str, interval: str = '1m', count: int = 1000):
    loader = DerivLoader()
    return loader.fetch_data(symbol, interval, count)
