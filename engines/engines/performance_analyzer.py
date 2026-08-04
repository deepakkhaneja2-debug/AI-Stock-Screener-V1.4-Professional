import pandas as pd
import os
from typing import Dict, Any


class PerformanceAnalyzer:
    """Enhanced performance analysis with monthly P&L and metrics."""

    def __init__(self, file_name: str = "trade_history.csv"):
        self.file_name = file_name

    def _load(self) -> pd.DataFrame:
        if not os.path.exists(self.file_name):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.file_name)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()

    def summary(self) -> Dict[str, Any]:
        df = self._load()
        if df.empty:
            return {
                "TotalTrades": 0,
                "ClosedTrades": 0,
                "Wins": 0,
                "Losses": 0,
                "BreakEven": 0,
                "WinRate": 0,
                "TotalPnL": 0,
                "AverageProfit": 0,
                "AverageLoss": 0,
                "ProfitFactor": 0,
                "Expectancy": 0
            }

        closed = df[df["Exit"] > 0].copy()
        total = len(closed)
        wins = len(closed[closed["PnL"] > 0])
        losses = len(closed[closed["PnL"] < 0])
        break_even = len(closed[closed["PnL"] == 0])

        winrate = round((wins / total) * 100, 2) if total > 0 else 0
        total_pnl = closed["PnL"].sum()

        avg_profit = closed[closed["PnL"] > 0]["PnL"].mean() if wins > 0 else 0
        avg_loss = closed[closed["PnL"] < 0]["PnL"].mean() if losses > 0 else 0

        gross_profit = closed[closed["PnL"] > 0]["PnL"].sum()
        gross_loss = abs(closed[closed["PnL"] < 0]["PnL"].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

        expectancy = round((wins/total * avg_profit) + (losses/total * avg_loss), 2) if total > 0 else 0

        return {
            "TotalTrades": total,
            "ClosedTrades": total,
            "Wins": wins,
            "Losses": losses,
            "BreakEven": break_even,
            "WinRate": winrate,
            "TotalPnL": round(total_pnl, 2),
            "AverageProfit": round(avg_profit, 2),
            "AverageLoss": round(avg_loss, 2),
            "ProfitFactor": profit_factor,
            "Expectancy": expectancy
        }

    def average_profit(self) -> float:
        df = self._load()
        if df.empty:
            return 0.0
        closed = df[df["Exit"] > 0]
        profit = closed[closed["PnL"] > 0]
        return round(profit["PnL"].mean(), 2) if not profit.empty else 0.0

    def average_loss(self) -> float:
        df = self._load()
        if df.empty:
            return 0.0
        closed = df[df["Exit"] > 0]
        loss = closed[closed["PnL"] < 0]
        return round(loss["PnL"].mean(), 2) if not loss.empty else 0.0

    def equity_curve(self) -> pd.DataFrame:
        df = self._load()
        if df.empty:
            return pd.DataFrame()
        closed = df[df["Exit"] > 0].sort_values("Date")
        if closed.empty:
            return pd.DataFrame()
        closed["Equity"] = closed["PnL"].cumsum()
        return closed[["Date", "Equity"]]

    def drawdown_curve(self) -> pd.DataFrame:
        df = self._load()
        if df.empty:
            return pd.DataFrame()
        closed = df[df["Exit"] > 0].sort_values("Date")
        if closed.empty:
            return pd.DataFrame()
        closed["Equity"] = closed["PnL"].cumsum()
        closed["Peak"] = closed["Equity"].cummax()
        closed["Drawdown"] = closed["Equity"] - closed["Peak"]
        return closed[["Date", "Drawdown"]]