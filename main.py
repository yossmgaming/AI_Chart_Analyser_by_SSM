import argparse
from signal_generator import generate_signals
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Trading Suite CLI")
    parser.add_argument("--symbols", type=str, default="AAPL,BTC-USD,TSLA,ETH-USD", help="Comma-separated list of symbols")
    args = parser.parse_args()

    symbols = args.symbols.split(',')

    print(f"\n{'Symbol':<10} | {'Signal':<15} | {'1h':<10} | {'1d':<10} | {'Price':<10}")
    print("-" * 65)

    results = []
    for symbol in symbols:
        signal_data = generate_signals(symbol.strip(), balance=1000, risk_pct=1)
        if isinstance(signal_data, dict):
            print(f"{signal_data['Symbol']:<10} | {signal_data['Final Signal']:<15} | {signal_data['1h Verdict']:<10} | {signal_data['1d Verdict']:<10} | {signal_data['Current Price']:<10}")
            results.append(signal_data)
        else:
            print(f"{symbol:<10} | Error: {signal_data}")

    # Optionally export to CSV
    # pd.DataFrame(results).to_csv("signals.csv", index=False)

if __name__ == "__main__":
    main()
