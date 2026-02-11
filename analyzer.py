import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RSI, MACD, Bollinger Bands, EMAs, ATR, ADX, and Ichimoku."""
    df['RSI'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df['Close'])
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)

    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])

    adx = ta.adx(df['High'], df['Low'], df['Close'])
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    try:
        ichi_df, _ = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichi_df is not None:
            df = pd.concat([df, ichi_df], axis=1)
    except:
        pass

    return df

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects all available candlestick patterns using pandas-ta."""
    patterns = df.ta.cdl_pattern(name="all")
    df['Detected_Patterns'] = ""
    for col in patterns.columns:
        mask = patterns[col] != 0
        # pandas-ta returns 100 for bullish, -100 for bearish patterns
        df.loc[mask, 'Detected_Patterns'] += col + " (" + patterns[col].astype(str) + "), "
    df['Detected_Patterns'] = df['Detected_Patterns'].str.rstrip(', ')
    return df

def detect_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Detects Bullish and Bearish RSI Divergence."""
    df['Bullish_Divergence'] = False
    df['Bearish_Divergence'] = False

    if 'RSI' not in df.columns or df['RSI'].isnull().all():
        return df

    rsi = df['RSI'].dropna().values
    close = df['Close'].iloc[len(df)-len(rsi):].values
    indices = df.index[len(df)-len(rsi):]

    # Bullish Divergence: Price lower low, RSI higher low
    minima, _ = find_peaks(-rsi, distance=5)
    for i in range(1, len(minima)):
        p, c = minima[i-1], minima[i]
        if close[c] < close[p] and rsi[c] > rsi[p]:
            df.at[indices[c], 'Bullish_Divergence'] = True

    # Bearish Divergence: Price higher high, RSI lower high
    maxima, _ = find_peaks(rsi, distance=5)
    for i in range(1, len(maxima)):
        p, c = maxima[i-1], maxima[i]
        if close[c] > close[p] and rsi[c] < rsi[p]:
            df.at[indices[c], 'Bearish_Divergence'] = True

    return df

def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> list:
    """Identifies major support and resistance levels."""
    levels = []
    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            levels.append(df['High'].iloc[i])
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            levels.append(df['Low'].iloc[i])

    # Simplified cleanup: Keep unique-ish levels
    if not levels:
        return []

    levels = sorted(list(set(levels)))
    cleaned_levels = []
    if levels:
        cleaned_levels.append(levels[0])
        for i in range(1, len(levels)):
            if levels[i] > cleaned_levels[-1] * 1.02: # 2% difference
                cleaned_levels.append(levels[i])
    return cleaned_levels

def confirm_signals(df: pd.DataFrame) -> str:
    """
    Returns a Buy/Sell/Hold verdict based on the latest data.
    """
    if df.empty:
        return "Hold"

    latest = df.iloc[-1]

    score = 0

    # RSI check
    if 'RSI' in latest and not pd.isna(latest['RSI']):
        if latest['RSI'] < 30: score += 1
        if latest['RSI'] > 70: score -= 1

    # MACD check
    macd_col = 'MACD_12_26_9'
    signal_col = 'MACDs_12_26_9'
    if macd_col in latest and signal_col in latest:
        if not pd.isna(latest[macd_col]) and not pd.isna(latest[signal_col]):
            if latest[macd_col] > latest[signal_col]: score += 1
            else: score -= 1

    # EMA Trend check
    if 'EMA_50' in latest and not pd.isna(latest['EMA_50']):
        if latest['Close'] > latest['EMA_50']: score += 1
        else: score -= 1

    # Divergence
    if 'Bullish_Divergence' in latest and latest['Bullish_Divergence']: score += 2
    if 'Bearish_Divergence' in latest and latest['Bearish_Divergence']: score -= 2

    if score >= 2: return "Buy"
    elif score <= -2: return "Sell"
    else: return "Hold"

def calculate_trade_levels(df: pd.DataFrame, verdict: str, rr_ratio: float = 2.0):
    """
    Calculates Entry, Stop Loss, and Take Profit levels using ATR.
    """
    if df.empty or verdict == "Hold":
        return None

    latest = df.iloc[-1]
    entry_price = latest['Close']
    atr = latest['ATR'] if 'ATR' in latest and not pd.isna(latest['ATR']) else entry_price * 0.01

    # Use 2x ATR for Stop Loss as a standard practice
    sl_dist = 2 * atr

    if verdict == "Buy":
        sl = entry_price - sl_dist
        tp = entry_price + (sl_dist * rr_ratio)
    else: # Sell
        sl = entry_price + sl_dist
        tp = entry_price - (sl_dist * rr_ratio)

    return {
        'Entry': round(entry_price, 5),
        'Stop Loss': round(sl, 5),
        'Take Profit': round(tp, 5),
        'ATR': round(atr, 5)
    }
