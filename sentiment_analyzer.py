import google.generativeai as genai
import os
import pandas as pd
import yfinance as yf
import json

class SentimentAnalyzer:
    def __init__(self, model_name='models/gemini-2.5-flash'):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def get_news(self, symbol):
        ticker = yf.Ticker(symbol)
        return ticker.news

    def analyze_sentiment(self, news_items):
        if not self.model:
            return [{"sentiment_score": 3, "risk_score": 3, "summary": "AI not configured"}] * len(news_items)

        results = []
        for item in news_items[:5]: # Analyze last 5 news items
            title = item.get('title', '')
            summary = item.get('summary', '')
            prompt = f"""
            Analyze the following financial news item and provide a structured JSON response.
            Title: {title}
            Summary: {summary}

            Return JSON with:
            - sentiment_score: 1 (very bearish) to 5 (very bullish)
            - risk_score: 1 (low risk/stable) to 5 (high risk/volatile)
            - justification: one sentence explanation
            """
            try:
                import time
                time.sleep(1) # Add delay to avoid rate limits
                response = self.model.generate_content(prompt)
                # Extract JSON from response (sometimes Gemini adds markdown code blocks)
                text = response.text.strip()
                if text.startswith('```json'):
                    text = text[7:-3].strip()
                elif text.startswith('```'):
                    text = text[3:-3].strip()

                # Clean up response text to find JSON
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)

                analysis = json.loads(text)
                # Ensure scores are numeric
                analysis['sentiment_score'] = int(analysis.get('sentiment_score', 3))
                analysis['risk_score'] = int(analysis.get('risk_score', 3))
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing news: {e}")
                results.append({"sentiment_score": 3, "risk_score": 3, "justification": "Analysis failed"})

        return results

    def get_aggregate_scores(self, symbol):
        news = self.get_news(symbol)
        if not news:
            return 3.0, 3.0 # Neutral

        analyses = self.analyze_sentiment(news)
        sentiments = [a.get('sentiment_score', 3) for a in analyses]
        risks = [a.get('risk_score', 3) for a in analyses]

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 3.0
        avg_risk = sum(risks) / len(risks) if risks else 3.0

        return avg_sentiment, avg_risk

if __name__ == "__main__":
    # Test
    analyzer = SentimentAnalyzer()
    s, r = analyzer.get_aggregate_scores("AAPL")
    print(f"AAPL Sentiment: {s}, Risk: {r}")
