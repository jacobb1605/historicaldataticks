# XAUUSD Modular Backtesting Engine (Tick + M1)

This project provides a modular, production-style Python backtesting engine for XAUUSD using:

- **Tick data** (bid/ask) or **1-minute OHLCV** bars
- ICT-style **market structure** (swings + BOS/MSS proxy)
- **FVG** / **IFVG** detection and testing
- **Fibonacci retracement** confluence setups
- No-lookahead engine loop, session filters, spread/slippage, and fixed-fractional risk sizing
- In-sample optimization separated from out-of-sample validation

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Data expectations

The loader is designed for multiple CSV files. It supports:

- **Tick CSV**: requires at least `time`, `bid`, `ask` columns (case-insensitive). Volume is optional.
- **M1 OHLCV CSV**: requires at least `time`, `open`, `high`, `low`, `close` columns. Volume optional.

Timestamps are parsed and normalized to timezone-aware UTC.

## Run

After you point `main_backtest.py` at your data folder(s):

```bash
python main_backtest.py --help
python main_backtest.py --m1_dir "C:\path\to\m1_csvs" --tick_dir "C:\path\to\tick_csvs"
```

The script will:
- Build an in-sample parameter sweep
- Select the best configuration(s) on in-sample
- Run them on out-of-sample
- Print a summary table and save trade/equity outputs

## Strategy documentation

- See `[docs/strategy_rules.md](docs/strategy_rules.md)` for the exact rule definitions.\n
- See `[docs/robustness_suggestions.md](docs/robustness_suggestions.md)` for recommended next steps to harden results.

