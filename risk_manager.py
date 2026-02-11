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

if __name__ == "__main__":
    rm = RiskManager()
    mock_returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 1)))
    turb = rm.calculate_turbulence(mock_returns)
    print(f"Current Turbulence: {turb}")
    kill, msg = rm.check_kill_switch(turb)
    print(msg)
