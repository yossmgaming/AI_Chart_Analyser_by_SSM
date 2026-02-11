# Professional Quantitative Trading Suite - Quant Upgrade

This upgrade transforms the trading suite into an advanced algorithmic system based on Multi-Agent Reinforcement Learning (MARL) and Deep Learning.

## Key Features
- **DeepLOB Architecture**: Hybrid CNN-LSTM model designed to capture spatial structures of the Limit Order Book (LOB).
- **MARL Ensemble**: Strategic ensemble of PPO and A2C agents that switch based on market regime performance.
- **Endogenous Modeling**: Integration with ABIDES simulation concepts for price formation analysis.
- **Real-time LOB**: Sub-second Level 2 data integration from Coinbase/Binance.
- **AI Sentiment**: News sentiment scoring using Gemini-2.5-flash.
- **Explainable AI (XAI)**: LIME-based decision transparency.
- **Risk Management**: Automated kill-switch based on the Financial Turbulence Index.

## New Modules
- `models.py`: Core DeepLOB implementation.
- `marl_agents.py`: Gymnasium environment and RL Ensemble logic.
- `exchange_l2_loader.py`: Real-time Order Book fetcher.
- `sentiment_analyzer.py`: LLM-based sentiment scoring.
- `risk_manager.py`: Turbulence and VaR/CVaR calculations.
- `exporter.py`: PDF and JSON reporting.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set your API Key: `export GOOGLE_API_KEY=your_key`
3. Run the GUI: `streamlit run app.py`

## ABIDES Integration
The system is compatible with the ABIDES-JPMC framework. To run full endogenous simulations:
1. Clone ABIDES: `git clone https://github.com/jpmorganchase/abides-jpmc-public.git`
2. Install ABIDES modules from the cloned directory.
3. Use the `abides_gym` environments as the base for `marl_agents.py`.
