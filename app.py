import streamlit as st
import pandas as pd
from data_loader import fetch_data
from deriv_loader import fetch_deriv_data
from analyzer import (
    add_technical_indicators,
    detect_candlestick_patterns,
    detect_divergence,
    detect_support_resistance,
    confirm_signals,
    calculate_trade_levels
)
from visualizer import create_chart
from backtester import run_backtest

st.set_page_config(layout="wide", page_title="Professional Trading Suite")

st.sidebar.title("Trading Suite Settings")

st.sidebar.header("Risk Management")
balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0)

data_source = st.sidebar.radio("Select Data Source", options=["Yahoo Finance", "Deriv"])

if data_source == "Yahoo Finance":
    symbol = st.sidebar.text_input("Enter Ticker Symbol", value="BTC-USD")
    interval = st.sidebar.selectbox("Select Interval", options=['1h', '1d', '1wk'], index=1)
    period = st.sidebar.selectbox("Select Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
else:
    deriv_symbols = {
        "Volatility 10 Index": "R_10",
        "Volatility 25 Index": "R_25",
        "Volatility 50 Index": "R_50",
        "Volatility 75 Index": "R_75",
        "Volatility 100 Index": "R_100",
        "Jump 10 Index": "JD10",
        "Jump 100 Index": "JD100",
        "Bear Market Index": "RDBEAR",
        "Bull Market Index": "RDBULL"
    }
    selected_name = st.sidebar.selectbox("Select Deriv Index", options=list(deriv_symbols.keys()))
    symbol = deriv_symbols[selected_name]
    interval = st.sidebar.selectbox("Select Interval", options=['1t', '1m', '5m', '15m', '1h', '1d'], index=1)
    count = st.sidebar.slider("Number of Data Points", min_value=100, max_value=5000, value=1000)

st.title(f"Market Analysis for {symbol}")

@st.cache_data(ttl=300)
def get_yahoo_data(symbol, interval, period):
    return fetch_data(symbol, interval=interval, period=period)

@st.cache_data(ttl=60)
def get_deriv_data(symbol, interval, count):
    return fetch_deriv_data(symbol, interval=interval, count=count)

try:
    # 1. Fetch Data
    if data_source == "Yahoo Finance":
        df = get_yahoo_data(symbol, interval, period)
    else:
        df = get_deriv_data(symbol, interval, count)

    # 2. Add Indicators & Patterns
    df = add_technical_indicators(df)
    df = detect_candlestick_patterns(df)
    df = detect_divergence(df)
    levels = detect_support_resistance(df)

    # 3. Sidebar Signal Info
    verdict = confirm_signals(df)
    st.sidebar.markdown(f"### Current Signal: **{verdict}**")

    latest_price = df['Close'].iloc[-1]
    st.sidebar.metric("Latest Price", f"{latest_price:.2f}")

    # Advanced Signal Dashboard
    st.header("Actionable Signal Details")
    if verdict != "Hold":
        trade_details = calculate_trade_levels(df, verdict)
        results, _ = run_backtest(df, initial_capital=balance, interval=interval)

        if trade_details and results:
            # Calculate Stake
            risk_amount = balance * (risk_pct / 100)
            price_diff = abs(trade_details['Entry'] - trade_details['Stop Loss'])
            stake = risk_amount / price_diff if price_diff > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.success(f"**Type:** {verdict}")
            c2.info(f"**Stake/Size:** {stake:.4f} units")
            c3.warning(f"**Stop Loss:** {trade_details['Stop Loss']}")
            c4.success(f"**Take Profit:** {trade_details['Take Profit']}")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Historical Win Rate", results['Win Rate'])

            # Formatting time based on interval
            if 'd' in interval or 'wk' in interval:
                time_str = df.index[-1].strftime('%Y-%m-%d')
            else:
                time_str = df.index[-1].strftime('%H:%M:%S')

            sc2.metric("Signal Time", time_str)
            sc3.metric("ATR Volatility", trade_details['ATR'])
    else:
        st.write("No active signal. Waiting for market conditions to align.")

    # 4. Main Chart
    fig = create_chart(df, symbol, levels=levels)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Backtest Section
    st.header("Backtest Simulation")
    if st.button("Run Backtest (RSI Strategy)"):
        results, backtest_df = run_backtest(df, interval=interval)
        if results:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", results['Total Return'])
            col2.metric("Buy & Hold", results['Buy & Hold Return'])
            col3.metric("Max Drawdown", results['Max Drawdown'])
            col4.metric("Sharpe Ratio", results['Sharpe Ratio'])

            st.markdown(f"**Final Capital:** ${results['Final Capital']}")

    # 6. Detected Patterns Table
    st.header("Latest Detected Patterns")
    patterns_df = df[df['Detected_Patterns'] != ""].tail(10)[['Detected_Patterns']]
    if not patterns_df.empty:
        st.table(patterns_df)
    else:
        st.write("No recent candlestick patterns detected.")

except Exception as e:
    st.error(f"Error: {e}")
