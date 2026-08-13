# AI Stock Scanner V1.3

A comprehensive stock scanning and analysis system built with Streamlit.

## Features

- **Data Engine**: Fetches historical data using yfinance with parallel downloads.
- **Indicator Engine**: Computes EMA, RSI, MACD, ATR, VWAP, and trend/momentum scores.
- **Pattern Engine**: Detects candlestick patterns (engulfing, hammer, doji, etc.).
- **Strategy Engine**: Combines 7 strategies (EMA, MACD, RSI, Volume Spike, Patterns, Breakout, Trend Score) to generate BUY/SELL/WATCH signals with a score.
- **Confidence Engine**: Computes a realistic confidence score using weighted contributions from strategy, trend, pattern, volume, and ATR.
- **Risk Engine**: Provides entry, stop-loss, targets, and position sizing based on ATR.
- **Backtest Engine**: Runs historical backtests with trade management and detailed analysis.
- **Dashboard**: Displays top BUY/SELL signals, summary stats, and per-symbol backtest results.
- **Alerts**: Streamlit notifications (Telegram/WhatsApp placeholders for future).
- **Trade Logger**: Saves all signals to CSV for performance tracking.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
