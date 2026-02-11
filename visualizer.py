import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_chart(df, symbol, levels=None, backtest_results=None):
    """
    Creates an interactive multi-panel chart with candlesticks and technical indicators.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.1, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price', '', 'RSI', 'MACD')
    )

    # 1. Main Price Chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # EMAs
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)
    if 'EMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'), row=1, col=1)
    if 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=1), name='EMA 200'), row=1, col=1)

    # Bollinger Bands
    if 'BBL_20_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dash'), name='Lower BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dash'), name='Upper BB'), row=1, col=1)

    # Ichimoku
    if 'ISA_9' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ISA_9'], line=dict(color='green', width=0.5), name='Senkou A'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ISB_26'], line=dict(color='red', width=0.5), name='Senkou B', fill='tonexty'), row=1, col=1)

    # Support/Resistance Levels
    if levels:
        for level in levels:
            fig.add_hline(y=level, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)

    # 2. Volume (Row 2)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'), row=2, col=1)

    # 3. RSI (Row 3)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Divergence markers
        bullish_div = df[df['Bullish_Divergence']]
        bearish_div = df[df['Bearish_Divergence']]
        fig.add_trace(go.Scatter(x=bullish_div.index, y=bullish_div['RSI'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Bullish Div'), row=3, col=1)
        fig.add_trace(go.Scatter(x=bearish_div.index, y=bearish_div['RSI'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Bearish Div'), row=3, col=1)

    # 4. MACD (Row 4)
    macd_col = 'MACD_12_26_9'
    signal_col = 'MACDs_12_26_9'
    hist_col = 'MACDh_12_26_9'
    if macd_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_col], name='MACD', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[signal_col], name='Signal', line=dict(color='orange')), row=4, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df[hist_col], name='Histogram'), row=4, col=1)

    fig.update_layout(height=1000, template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig
