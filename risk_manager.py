import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, turbulence_threshold=50.0):
        self.turbulence_threshold = turbulence_threshold

    def calculate_turbulence(self, returns_df):
        """
        Computes the Financial Turbulence Index.
        returns_df: DataFrame of asset returns.
        """
        if returns_df.empty:
            return 0.0

        mu = returns_df.mean()
        sigma = returns_df.cov()

        try:
            sigma_inv = np.linalg.inv(sigma)
            # Latest return
            y = returns_df.iloc[-1].values
            delta = y - mu.values
            turbulence = delta.dot(sigma_inv).dot(delta.T)
            return turbulence
        except:
            # Fallback for single asset or singular matrix
            r = returns_df.iloc[-1].values[0]
            m = mu.values[0]
            v = sigma.values[0][0]
            if v == 0: return 0.0
            return ((r - m)**2) / v

    def check_kill_switch(self, current_turbulence):
        if current_turbulence > self.turbulence_threshold:
            return True, "Kill-switch activated: Financial Turbulence too high!"
        return False, "Market stable."

    def calculate_var_cvar(self, returns, confidence=0.95):
        """
        Calculates Value-at-Risk and Conditional Value-at-Risk.
        """
        if len(returns) < 10:
            return 0.0, 0.0

        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = abs(sorted_returns[index])
        cvar = abs(sorted_returns[:index].mean())
        return var, cvar

    def get_adaptive_stake(self, balance, returns, risk_pct=0.01):
        """
        Suggests a stake size based on CVaR to protect low equity.
        """
        var, cvar = self.calculate_var_cvar(returns)
        if cvar == 0:
            return balance * risk_pct

        # If CVaR is high (e.g. 5% loss in tail), we reduce stake
        # Logic: Stake * CVaR should not exceed Balance * Risk_Pct
        suggested_stake = (balance * risk_pct) / cvar
        return min(suggested_stake, balance * 0.2) # Max 20% of balance

    def check_trading_session(self, current_time=None):
        """
        Returns whether the current time is optimal for trading.
        Generally, high volume sessions (NY/London) are better.
        """
        import pytz
        from datetime import datetime
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        hour = current_time.hour
        # London: 8-16 UTC, NY: 13-21 UTC
        # Overlap: 13-16 UTC (Best)
        is_london = 8 <= hour <= 16
        is_ny = 13 <= hour <= 21

        if is_london and is_ny:
            return True, "Peak Market Overlap (London & NY). High liquidity/volatility."
        elif is_london:
            return True, "London Session. Good liquidity."
        elif is_ny:
            return True, "NY Session. Good liquidity."
        else:
            return False, "Off-peak hours. Expect lower liquidity and potential chop."

    def calculate_daily_target(self, balance, target_pct=0.02):
        """
        Returns the daily profit target in currency.
        """
        return balance * target_pct

if __name__ == "__main__":
    rm = RiskManager()
    mock_returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 1)))
    turb = rm.calculate_turbulence(mock_returns)
    print(f"Current Turbulence: {turb}")
    kill, msg = rm.check_kill_switch(turb)
    print(msg)
