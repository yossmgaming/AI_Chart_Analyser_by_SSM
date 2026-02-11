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
    Fetches AI-driven trading suggestions.
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
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            As a professional trading assistant, analyze the following data for {symbol}:
            - Action: {verdict}
            - Current Price: {context['price']}
            - RSI: {context['rsi']}
            - Patterns: {context['patterns']}
            - Recommended Stop Loss: {context['sl']}
            - Recommended Take Profit: {context['tp']}

            Provide a brief strategy explanation, the primary risk for this trade, and one 'pro deal' or tip for this specific setup.

            Context on Model Decision (LIME Explanations):
            {explanations if explanations else "No specific model data available."}

            Keep it concise and professional.
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
