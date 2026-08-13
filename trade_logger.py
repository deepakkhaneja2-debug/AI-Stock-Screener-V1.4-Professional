import os
import pandas as pd
from datetime import datetime


class TradeLogger:
    """Logs trades to a CSV file."""

    def __init__(self, file_name: str = "trade_history.csv"):
        self.file_name = file_name
        if not os.path.exists(self.file_name):
            df = pd.DataFrame(columns=[
                "Date", "Time", "Symbol", "Signal", "Entry", "Exit",
                "StopLoss", "Target", "Quantity", "PnL", "PnLPercent",
                "Reason", "Confidence", "EMA", "MACD", "RSI", "Pattern", "TrendScore"
            ])
            df.to_csv(self.file_name, index=False)

    def save_trade(self, symbol: str, signal: str, entry: float, exit_price: float,
                   sl: float, target: float, qty: int, pnl: float, pnl_percent: float,
                   reason: str, confidence: float, ema: float, macd: float, rsi: float,
                   pattern: int, trend: int) -> None:
        """Save a trade to the CSV file."""
        try:
            df = pd.read_csv(self.file_name)
        except Exception:
            df = pd.DataFrame(columns=[
                "Date", "Time", "Symbol", "Signal", "Entry", "Exit",
                "StopLoss", "Target", "Quantity", "PnL", "PnLPercent",
                "Reason", "Confidence", "EMA", "MACD", "RSI", "Pattern", "TrendScore"
            ])

        now = datetime.now()
        new_row = pd.DataFrame([[
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            symbol, signal, entry, exit_price, sl, target, qty, pnl, pnl_percent,
            reason, confidence, ema, macd, rsi, pattern, trend
        ]], columns=df.columns)

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.file_name, index=False)

    def history(self) -> pd.DataFrame:
        """Return the entire trade history."""
        if not os.path.exists(self.file_name):
            return pd.DataFrame()
        return pd.read_csv(self.file_name)

    def total_trades(self) -> int:
        """Return total number of trades."""
        return len(self.history())

    def closed_trades(self) -> pd.DataFrame:
        """Return only closed trades (exit price > 0)."""
        df = self.history()
        if df.empty:
            return pd.DataFrame()
        return df[df["Exit"] > 0]