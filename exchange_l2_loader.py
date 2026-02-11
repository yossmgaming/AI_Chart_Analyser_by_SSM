import pandas as pd
import numpy as np
import requests
import time

class ExchangeL2Loader:
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"

    def get_order_book(self, symbol="BTC-USD", limit=50):
        """
        Fetches the current order book snapshot from Coinbase.
        """
        url = f"{self.base_url}/products/{symbol}/book?level=2"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Coinbase returns {bids: [[price, size, num_orders], ...], asks: [...]}
            return data
        else:
            raise Exception(f"Failed to fetch LOB: {response.status_code} {response.text}")

    def process_lob_to_feature(self, depth, levels=10):
        """
        Converts order book depth to a feature vector suitable for DeepLOB.
        """
        asks = depth.get('asks', [])[:levels]
        bids = depth.get('bids', [])[:levels]

        features = []
        for i in range(levels):
            # Price, Volume
            ap = float(asks[i][0]) if i < len(asks) else 0
            av = float(asks[i][1]) if i < len(asks) else 0
            bp = float(bids[i][0]) if i < len(bids) else 0
            bv = float(bids[i][1]) if i < len(bids) else 0
            features.extend([ap, av, bp, bv])

        return np.array(features)

if __name__ == "__main__":
    loader = ExchangeL2Loader()
    depth = loader.get_order_book("BTC-USD", limit=10)
    features = loader.process_lob_to_feature(depth, levels=10)
    print(f"LOB Features shape: {features.shape}")
    print(f"Sample (Top Ask): Price {features[0]}, Vol {features[1]}")
