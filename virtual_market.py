import numpy as np
import pandas as pd
from collections import deque

class VirtualMarket:
    """
    A simplified simulation environment inspired by ABIDES-MARL.
    Features:
    - Endogenous price formation (agent trades impact LOB).
    - Limit Order Book (LOB) with multiple levels.
    - Synthetic noise traders and market makers.
    """
    def __init__(self, symbol="SIM-USD", levels=10, initial_price=60000.0):
        self.symbol = symbol
        self.levels = levels
        self.mid_price = initial_price
        self.spread = 0.01 # 0.01% spread

        # LOB: List of [Price, Volume]
        self.bids = []
        self.asks = []
        self._seed_lob()

        self.history = deque(maxlen=1000)
        self.agent_positions = [] # Track open trades

    def _seed_lob(self):
        """Initializes the LOB with synthetic liquidity."""
        self.bids = []
        self.asks = []
        for i in range(self.levels):
            bid_p = self.mid_price * (1 - (self.spread/2) - (i * 0.0005))
            ask_p = self.mid_price * (1 + (self.spread/2) + (i * 0.0005))
            self.bids.append([round(bid_p, 2), np.random.uniform(0.1, 2.0)])
            self.asks.append([round(ask_p, 2), np.random.uniform(0.1, 2.0)])

    def get_lob_snapshot(self):
        return {
            'bids': sorted(self.bids, key=lambda x: x[0], reverse=True),
            'asks': sorted(self.asks, key=lambda x: x[0])
        }

    def process_agent_order(self, side, volume):
        """
        Processes an agent's market order and calculates impact/slippage.
        Returns execution price and implementation shortfall.
        """
        snapshot = self.get_lob_snapshot()
        mid_before = (snapshot['asks'][0][0] + snapshot['bids'][0][0]) / 2

        executed_vol = 0
        total_cost = 0

        if side == 'buy':
            target_list = self.asks
        else:
            target_list = self.bids

        # Match against LOB
        for i in range(len(target_list)):
            price, vol = target_list[i]
            if volume <= executed_vol:
                break

            fill = min(volume - executed_vol, vol)
            total_cost += fill * price
            executed_vol += fill
            target_list[i][1] -= fill

        execution_price = total_cost / executed_vol if executed_vol > 0 else mid_before
        shortfall = abs(execution_price - mid_before) / mid_before

        # Endogenous impact: Update mid price
        impact_factor = 0.0001 * (volume / 1.0)
        if side == 'buy':
            self.mid_price *= (1 + impact_factor)
        else:
            self.mid_price *= (1 - impact_factor)

        self._update_market()
        return execution_price, shortfall

    def _update_market(self):
        """Simulates noise traders and market makers replenishing liquidity."""
        # Refresh book around new mid price
        self._seed_lob()
        # Add random noise
        for b in self.bids: b[1] += np.random.uniform(-0.05, 0.1)
        for a in self.asks: a[1] += np.random.uniform(-0.05, 0.1)

        self.history.append(self.mid_price)

    def step(self):
        """Standard simulation step with random noise."""
        noise = np.random.normal(0, 0.0002) # 0.02% vol
        self.mid_price *= (1 + noise)
        self._update_market()
        return self.mid_price

if __name__ == "__main__":
    market = VirtualMarket()
    print("Initial LOB:", market.get_lob_snapshot()['asks'][0])
    price, shortfall = market.process_agent_order('buy', 0.5)
    print(f"Executed at {price}, Shortfall: {shortfall:.6f}")
    print("New Mid:", market.mid_price)
