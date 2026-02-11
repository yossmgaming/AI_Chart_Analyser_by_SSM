from data_loader import fetch_data
from analyzer import add_technical_indicators, detect_divergence, confirm_signals

def generate_signals(symbol: str):
    """
    Generates multi-timeframe signals for a given symbol.
    """
    # Fetch 1h and 1d data
    try:
        # Fetch 2 years for daily to ensure long-term EMAs (like 200) are calculated
        df_1h, df_1d = fetch_data(symbol, interval='1h', period='2y', fetch_daily=True)
    except Exception as e:
        return f"Error fetching data for {symbol}: {e}"

    # Analyze both timeframes
    df_1h = add_technical_indicators(df_1h)
    df_1h = detect_divergence(df_1h)

    df_1d = add_technical_indicators(df_1d)
    df_1d = detect_divergence(df_1d)

    # Get verdicts
    verdict_1h = confirm_signals(df_1h)
    verdict_1d = confirm_signals(df_1d)

    # Combine signals
    if verdict_1h == "Buy" and verdict_1d == "Buy":
        final_signal = "Strong Buy"
    elif verdict_1h == "Sell" and verdict_1d == "Sell":
        final_signal = "Strong Sell"
    elif verdict_1h == "Buy" or verdict_1d == "Buy":
        final_signal = "Weak Buy"
    elif verdict_1h == "Sell" or verdict_1d == "Sell":
        final_signal = "Weak Sell"
    else:
        final_signal = "Neutral"

    return {
        'Symbol': symbol,
        'Final Signal': final_signal,
        '1h Verdict': verdict_1h,
        '1d Verdict': verdict_1d,
        'Current Price': f"{df_1h['Close'].iloc[-1]:.2f}"
    }
