import google.generativeai as genai
import os
import numpy as np
from lime import lime_tabular

class ExplainableAI:
    def __init__(self, agent, feature_names):
        self.agent = agent
        self.feature_names = feature_names
        self.explainer = None

    def predict_fn(self, x):
        # SB3 predict returns (action, states)
        # We need to return a probability-like distribution for classes
        # For simplicity, we'll mock the probabilities or use the policy if accessible
        # Since we use Discrete(3), we want (N, 3)
        results = []
        for obs in x:
            action, _ = self.agent.predict(obs, deterministic=True)
            # Mocking probability for LIME (1.0 for predicted action)
            prob = np.zeros(3)
            prob[action] = 1.0
            results.append(prob)
        return np.array(results)

    def explain_trade(self, observation):
        if self.explainer is None:
            # We initialize with some training data if available, or just random sample
            train_data = np.random.normal(0, 1, (100, len(self.feature_names)))
            self.explainer = lime_tabular.LimeTabularExplainer(
                train_data,
                feature_names=self.feature_names,
                class_names=['Hold', 'Buy', 'Sell'],
                mode='classification'
            )

        exp = self.explainer.explain_instance(observation, self.predict_fn, num_features=5)
        return exp.as_list()

def get_ai_suggestions(df, symbol, verdict, trade_details=None, explanations=None):
    """
    Fetches AI-driven trading suggestions as a Master Consultant.
    Attempts to use Google Gemini if API_KEY is present, otherwise uses rule-based logic.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    # Prepare context
    latest = df.iloc[-1]
    context = {
        "symbol": symbol,
        "verdict": verdict,
        "price": latest['Close'],
        "rsi": latest['RSI'],
        "patterns": latest['Detected_Patterns'],
        "sl": trade_details['Stop Loss'] if trade_details else "N/A",
        "tp": trade_details['Take Profit'] if trade_details else "N/A"
    }

    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Use 1.5-flash or 2.0-flash if possible, fallback to pro
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            You are a Master Quant Consultant. Analyze the following data for {symbol}:
            - Action: {verdict}
            - Current Price: {context['price']}
            - RSI: {context['rsi']}
            - Patterns: {context['patterns']}
            - Recommended Stop Loss: {context['sl']}
            - Recommended Take Profit: {context['tp']}

            Your goal is to provide a Strategic Order Blueprint.
            Explain why the DeepLOB model and MARL agents reached this conclusion.
            Highlight the 'Informational Advantage' we have over manual traders.

            Context on Model Decision (LIME Explanations):
            {explanations if explanations else "No specific model data available."}

            Be authoritative, professional, and concise.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Advisor Error: {e}. Falling back to rule-based analysis."

    # Fallback Rule-Based Suggestions
    suggestions = f"### Strategy Advisor (Rule-Based)\n\n"

    if verdict == "Buy":
        suggestions += f"**Strategy:** Look for a bullish continuation. RSI at {context['rsi']:.2f} suggests "
        suggestions += "room for growth." if context['rsi'] < 60 else "potential overbought conditions soon; be cautious."
        suggestions += f"\n\n**Risk:** Failure to break immediate resistance or negative divergence on lower timeframes."
    elif verdict == "Sell":
        suggestions += f"**Strategy:** Capitilize on bearish momentum. RSI at {context['rsi']:.2f} indicates "
        suggestions += "strong selling pressure." if context['rsi'] > 40 else "potential oversold bounce; use tight stops."
        suggestions += f"\n\n**Risk:** Sudden short-squeeze or bullish reversal patterns forming at support."
    else:
        suggestions += "**Strategy:** Market is in neutral/consolidation phase. Best to stay flat and wait for a clear breakout."
        suggestions += f"\n\n**Risk:** High 'chop' risk. Entering now may lead to being stopped out by noise."

    suggestions += "\n\n**Special Deal/Opportunity:** "
    if "Bull" in context.get('patterns', ''):
        suggestions += "Bullish pattern detected! This is a high-probability 'Buy the Dip' deal."
    elif "Bear" in context.get('patterns', ''):
        suggestions += "Bearish pattern spotted. Great deal for short-sellers or hedging existing longs."
    else:
        suggestions += "Market is quiet. The best 'deal' is patience—save your capital for the next big breakout."

    suggestions += "\n\n**Pro Tip:** Always check the economic calendar for high-impact news before executing."

    return suggestions

def get_strategic_directive(df, symbol, balance, risk_pct, verdict, trade_details):
    """
    Constructs the specific 'Buy at X, Sell after N minutes' directive requested.
    """
    from risk_manager import RiskManager
    rm = RiskManager()

    daily_target = rm.calculate_daily_target(balance)
    session_ok, session_msg = rm.check_trading_session()

    returns = df['Close'].pct_change().dropna()
    stake = rm.get_adaptive_stake(balance, returns, risk_pct)

    latest = df.iloc[-1]
    price = latest['Close']

    # Heuristic for duration: Based on ATR / Avg return per period
    # If ATR is 1% and we want 2% profit, it might take 2 periods.
    atr = latest['ATR'] if 'ATR' in latest else price * 0.01
    target_profit = abs(trade_details['Take Profit'] - trade_details['Entry']) if trade_details else atr * 2

    # Avg candle size (High-Low)
    avg_volatility = (df['High'] - df['Low']).tail(20).mean()
    if avg_volatility == 0: avg_volatility = atr

    estimated_periods = max(1, int(target_profit / avg_volatility))

    # Convert periods to minutes
    # We need to know the interval. Let's assume 1m for this calculation if not specified,
    # but ideally we get it from the dataframe frequency.
    interval_min = 1
    if hasattr(df.index, 'freq') and df.index.freq:
        interval_min = df.index.freq.delta.total_seconds() / 60
    else:
        # Infer from index
        if len(df) > 1:
            interval_min = (df.index[1] - df.index[0]).total_seconds() / 60

    estimated_minutes = int(estimated_periods * interval_min)

    directive = {
        "Strategic Daily Target": f"${daily_target:.2f}",
        "Optimized Stake": f"${stake:.2f}",
        "Action": verdict,
        "Entry Price": f"{trade_details['Entry']:.5f}" if trade_details else f"{price:.5f}",
        "Target Duration": f"{estimated_minutes} minutes",
        "Expected Profit": f"${(stake * (target_profit/trade_details['Entry'] if trade_details else 0)):.2f}" if trade_details else "TBD",
        "Session Status": session_msg
    }

    return directive
