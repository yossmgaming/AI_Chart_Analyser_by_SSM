import streamlit as st
import pandas as pd
from data_loader import fetch_data
from analyzer import (
    add_technical_indicators,
    detect_candlestick_patterns,
    detect_divergence,
    detect_support_resistance,
    confirm_signals
)
from visualizer import create_chart
from backtester import run_backtest

st.set_page_config(layout="wide", page_title="Professional Trading Suite")

st.sidebar.title("Trading Suite Settings")
symbol = st.sidebar.text_input("Enter Ticker Symbol", value="BTC-USD")
interval = st.sidebar.selectbox("Select Interval", options=['1h', '1d', '1wk'], index=1)
period = st.sidebar.selectbox("Select Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)

st.title(f"Market Analysis for {symbol}")

try:
    # 1. Fetch Data
    df = fetch_data(symbol, interval=interval, period=period)

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

    # 4. Main Chart
    fig = create_chart(df, symbol, levels=levels)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Backtest Section
    st.header("Backtest Simulation")
    if st.button("Run Backtest (RSI Strategy)"):
        results, backtest_df = run_backtest(df)
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
