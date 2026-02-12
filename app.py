import streamlit as st
import pandas as pd
import pytz
import numpy as np
from datetime import datetime
from data_loader import fetch_data, get_available_sources
from deriv_loader import fetch_deriv_data
from exchange_l2_loader import ExchangeL2Loader
from sentiment_analyzer import SentimentAnalyzer
from risk_manager import RiskManager
from analyzer import (
    add_technical_indicators,
    detect_candlestick_patterns,
    detect_divergence,
    detect_support_resistance,
    confirm_signals,
    calculate_trade_levels,
    calculate_turbulence
)
from virtual_market import VirtualMarket
from marl_agents import MultiAgentTradingEnv, TradingEnsemble
from visualizer import create_chart
from backtester import run_backtest
from ai_advisor import get_ai_suggestions, ExplainableAI, get_strategic_directive
from exporter import export_to_pdf, export_to_json
import plotly.express as px
import time
import os

st.set_page_config(layout="wide", page_title="Universal AI Trading Suite")

# --- Initializations ---
if 'market' not in st.session_state:
    st.session_state.market = VirtualMarket()
if 'env' not in st.session_state:
    st.session_state.env = MultiAgentTradingEnv(st.session_state.market)
if 'ensemble' not in st.session_state:
    st.session_state.ensemble = TradingEnsemble(st.session_state.env)

# --- Sidebar ---
st.sidebar.title("Trading Suite Settings")

st.sidebar.header("Regional Settings")
tz_list = pytz.all_timezones
default_tz_idx = tz_list.index("UTC") if "UTC" in tz_list else 0
selected_tz = st.sidebar.selectbox("Select Timezone", options=tz_list, index=default_tz_idx)
current_time_obj = datetime.now(pytz.timezone(selected_tz))
current_time_str = current_time_obj.strftime("%Y-%m-%d %H:%M:%S")

st.sidebar.header("Risk Management")
balance = st.sidebar.number_input("Account Balance ($)", value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0)
daily_target_pct = st.sidebar.slider("Daily Profit Target (%)", min_value=0.5, max_value=10.0, value=2.0)

sources = get_available_sources()
data_source = st.sidebar.radio("Select Data Source", options=sources)

symbol = "BTC-USD"
interval = "1h"
period = "1y"
limit = 500

if data_source == 'yfinance':
    symbol = st.sidebar.text_input("Enter Ticker Symbol", value="BTC-USD")
    interval = st.sidebar.selectbox("Select Interval", options=['1m', '5m', '15m', '1h', '1d', '1wk'], index=3)
    period = st.sidebar.selectbox("Select Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
elif data_source == 'binance':
    symbol = st.sidebar.text_input("Enter Binance Symbol (e.g. BTCUSDT)", value="BTCUSDT")
    interval = st.sidebar.selectbox("Select Interval", options=['1m', '5m', '15m', '1h', '4h', '1d'], index=3)
    limit = st.sidebar.slider("Limit", 100, 1000, 500)
elif data_source == 'deriv':
    deriv_symbols = {"Volatility 100": "R_100", "Volatility 75": "R_75", "Gold": "frxXAUUSD"}
    symbol = st.sidebar.selectbox("Select Deriv Index", options=list(deriv_symbols.keys()))
    symbol = deriv_symbols[symbol]
    interval = st.sidebar.selectbox("Select Interval", options=['1m', '5m', '1h', '1d'], index=0)
    limit = st.sidebar.slider("Count", 100, 5000, 1000)

st.title(f"Market Analysis for {symbol}")
st.info(f"Time: **{current_time_str} ({selected_tz})** | Balance: **${balance:,.2f}** | Source: **{data_source}**")

rm = RiskManager()
session_ok, session_msg = rm.check_trading_session(current_time_obj)
st.sidebar.markdown(f"**Session:** {session_msg}")

@st.cache_data(ttl=60)
def get_cached_data(symbol, interval, period, source, limit):
    return fetch_data(symbol, interval=interval, period=period, source=source, limit=limit)

try:
    # 1. Fetch and Analyze Data
    if data_source == 'coinbase_l2':
        loader = ExchangeL2Loader()
        depth = loader.get_order_book(symbol)
        st.subheader("Real-time LOB (Top 10 Levels)")
        lob_df = pd.DataFrame({
            'Bid Price': [b[0] for b in depth['bids'][:10]],
            'Bid Vol': [b[1] for b in depth['bids'][:10]],
            'Ask Price': [a[0] for a in depth['asks'][:10]],
            'Ask Vol': [a[1] for a in depth['asks'][:10]]
        })
        st.table(lob_df)
        df = get_cached_data(symbol, "1h", "1mo", "yfinance", 500)
        # Prepare observation for RL
        lob_features = loader.process_lob_to_feature(depth)
        obs = np.concatenate([lob_features, [3.0], np.zeros(5)]).astype(np.float32)
    else:
        df = get_cached_data(symbol, interval, period, data_source, limit)
        # Mock observation for RL when LOB is not available
        obs = np.random.normal(0, 1, 46).astype(np.float32)

    df = add_technical_indicators(df)
    df = detect_candlestick_patterns(df)
    df = detect_divergence(df)
    levels = detect_support_resistance(df)

    # RL/MARL Verdict & Step Environment
    if st.button("Simulate Market Step (Tick)"):
        _, rew, _, _, _ = st.session_state.env.step(np.random.choice([0, 1, 2])) # Random action for demo
        st.session_state.market.step()

    prediction = st.session_state.ensemble.get_detailed_prediction(obs)
    verdicts = ["Hold", "Buy", "Sell"]
    verdict = verdicts[prediction['action']]

    trade_details = calculate_trade_levels(df, verdict)

    # 2. Strategic Execution Directive (Quant/AI)
    st.header("🎯 Strategic Execution Directive (MARL-Driven)")

    # Show active agent badge
    agent_col, dummy = st.columns([1, 4])
    with agent_col:
        st.success(f"**Agent:** {prediction['agent_name']} ACTIVE")

    directive = get_strategic_directive(df, symbol, balance, risk_pct/100, verdict, trade_details)
    directive["Win Rate"] = prediction['win_rate']
    directive["Horizon"] = prediction['duration']

    with st.container(border=True):
        cols = st.columns(len(directive))
        for i, (k, v) in enumerate(directive.items()):
            cols[i].metric(k, v)

    # DeepLOB Multi-Horizon Forecasts
    st.subheader("DeepLOB Multi-Horizon Forecasts")
    h_cols = st.columns(len(prediction['multi_horizon']))
    for i, (h, v) in enumerate(prediction['multi_horizon'].items()):
        h_cols[i].metric(f"k={h}", v)

    # Execution Logs (Implementation Shortfall)
    st.subheader("Live Execution Logs (Endogenous Market)")
    if st.session_state.env.execution_logs:
        log_df = pd.DataFrame(st.session_state.env.execution_logs).tail(5)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No active trades executed in this session.")

    # 3. Decision Transparency (XAI)
    if st.checkbox("🔍 Show Decision Transparency (XAI - LIME)"):
        with st.spinner("Generating XAI report..."):
            feat_names = [f"LOB_{i}" for i in range(40)] + ["Sentiment"] + ["EMA20", "EMA50", "EMA200", "RSI", "ATR"]
            xai = ExplainableAI(st.session_state.ensemble.agents[prediction['agent_name']], feat_names)
            exps = xai.explain_trade(obs)
            exp_df = pd.DataFrame(exps, columns=["Feature", "Influence"])
            exp_df['Direction'] = exp_df['Influence'].apply(lambda x: 'Supportive' if x > 0 else 'Opposing')
            fig_exp = px.bar(exp_df, x="Influence", y="Feature", orientation='h',
                             color='Direction',
                             color_discrete_map={'Supportive':'green', 'Opposing':'red'},
                             title=f"Feature Influence for {prediction['agent_name']} {verdict} Decision")
            st.plotly_chart(fig_exp)

    # 4. Market Intelligence
    st.header("Market Intelligence")
    col_ai, col_story = st.columns([1, 1])

    with col_ai:
        if st.button("Analyze Real-time Sentiment"):
            with st.spinner("Analyzing global sentiment via Gemini..."):
                sa = SentimentAnalyzer()
                sent, risk = sa.get_aggregate_scores(symbol)
                st.write(f"**Sentiment Score:** {sent:.2f}/5.0 | **Risk Index:** {risk:.2f}/5.0")

        ai_msg = get_ai_suggestions(df, symbol, verdict, trade_details)
        st.markdown(ai_msg)

    with col_story:
        rsi_val = df['RSI'].iloc[-1]
        atr_val = df['ATR'].iloc[-1]
        st.write(f"**Technical Pulse:** RSI at {rsi_val:.2f}, ATR at {atr_val:.4f}")

        if interval != '1d':
            st.write("**Multi-Timeframe Status:** Checking Daily trend...")
            df_daily = get_cached_data(symbol, '1d', '1y', 'yfinance' if data_source in ['yfinance', 'coinbase_l2'] else data_source, 100)
            daily_trend = "Bullish" if df_daily['Close'].iloc[-1] > df_daily['Close'].rolling(50).mean().iloc[-1] else "Bearish"
            st.write(f"Daily Trend is **{daily_trend}**. Signal is **{'Confirmed' if (verdict == 'Buy' and daily_trend == 'Bullish') or (verdict == 'Sell' and daily_trend == 'Bearish') else 'Unconfirmed'}**.")

    # 5. Charting
    fig = create_chart(df, symbol, levels=levels)
    st.plotly_chart(fig, use_container_width=True)

    # 6. Performance & Backtest
    st.header("Strategy Performance")
    if st.button("Run Full Performance Simulation"):
        results, backtest_df = run_backtest(df, initial_capital=balance, interval=interval)
        if results:
            st.table(pd.DataFrame([results]))

            # Export
            report_data = {
                "Directive": directive,
                "Backtest Results": results,
                "Market Context": {"Symbol": symbol, "Price": df['Close'].iloc[-1]}
            }
            pdf_path = export_to_pdf(report_data)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Strategic Report (PDF)", f, file_name=f"{symbol}_report.pdf")

except Exception as e:
    st.error(f"System Error: {e}")
    st.exception(e)
