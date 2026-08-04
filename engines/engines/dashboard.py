import pandas as pd
from typing import Dict, Any

from config import *


class DashboardEngine:
    """Generates dashboard views and statistics."""

    def __init__(self):
        pass

    def top_buy(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return top BUY signals sorted by confidence."""
        if results.empty or "Signal" not in results.columns:
            return pd.DataFrame()
        buys = results[results["Signal"] == "BUY"].copy()
        if buys.empty:
            return pd.DataFrame()
        if "Confidence" in buys.columns:
            return buys.sort_values("Confidence", ascending=False).head(TOP_BUY_RESULTS)
        return buys.head(TOP_BUY_RESULTS)

    def top_sell(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return top SELL signals sorted by confidence."""
        if results.empty or "Signal" not in results.columns:
            return pd.DataFrame()
        sells = results[results["Signal"] == "SELL"].copy()
        if sells.empty:
            return pd.DataFrame()
        if "Confidence" in sells.columns:
            return sells.sort_values("Confidence", ascending=False).head(TOP_SELL_RESULTS)
        return sells.head(TOP_SELL_RESULTS)

    def summary(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        if results.empty:
            return {
                "Total Signals": 0,
                "Buys": 0,
                "Sells": 0,
                "Watches": 0,
                "Avg Confidence": 0
            }

        total = len(results)
        buys = len(results[results["Signal"] == "BUY"]) if "Signal" in results.columns else 0
        sells = len(results[results["Signal"] == "SELL"]) if "Signal" in results.columns else 0
        watches = len(results[results["Signal"] == "WATCH"]) if "Signal" in results.columns else 0

        avg_conf = results["Confidence"].mean() if "Confidence" in results.columns else 0

        return {
            "Total Signals": total,
            "Buys": buys,
            "Sells": sells,
            "Watches": watches,
            "Avg Confidence": round(avg_conf, 2)
        }

    def overall_stats(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics from backtest reports."""
        if not reports:
            return {
                "Total Trades": 0,
                "Closed Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "BreakEven": 0,
                "Win Rate": 0,
                "Total PnL": 0,
                "Profit Factor": 0,
                "AI Score": 0
            }

        all_trades = []
        ai_scores = []

        for symbol, report in reports.items():
            if not isinstance(report, dict):
                continue

            trades = report.get("Trades", [])
            if isinstance(trades, list):
                all_trades.extend(trades)

            ai_score = report.get("AI Score", 0)
            if isinstance(ai_score, (int, float)):
                ai_scores.append(ai_score)

        if not all_trades:
            return {
                "Total Trades": 0,
                "Closed Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "BreakEven": 0,
                "Win Rate": 0,
                "Total PnL": 0,
                "Profit Factor": 0,
                "AI Score": 0
            }

        total = len(all_trades)
        wins = sum(1 for t in all_trades if t.get("Status") == "WIN")
        losses = sum(1 for t in all_trades if t.get("Status") == "LOSS")
        break_even = sum(1 for t in all_trades if t.get("Status") == "BREAK_EVEN")
        closed = wins + losses + break_even

        win_rate = round((wins / closed) * 100, 2) if closed > 0 else 0

        total_pnl = round(sum(t.get("PnL", 0) for t in all_trades), 2)

        profits = [t.get("PnL", 0) for t in all_trades if t.get("Status") == "WIN"]
        losses_list = [t.get("PnL", 0) for t in all_trades if t.get("Status") == "LOSS"]

        gross_profit = sum(profits)
        gross_loss = abs(sum(losses_list))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

        avg_ai_score = round(sum(ai_scores) / len(ai_scores), 0) if ai_scores else 0

        return {
            "Total Trades": total,
            "Closed Trades": closed,
            "Wins": wins,
            "Losses": losses,
            "BreakEven": break_even,
            "Win Rate": win_rate,
            "Total PnL": total_pnl,
            "Profit Factor": profit_factor,
            "AI Score": avg_ai_score
        }

    def ranking_table(self, reports: Dict[str, Any]) -> pd.DataFrame:
        """Generate ranking table sorted by multiple metrics."""
        if not reports:
            return pd.DataFrame()

        rows = []
        for symbol, report in reports.items():
            if isinstance(report, dict):
                rows.append({
                    "Symbol": symbol,
                    "AI Score": report.get("AI Score", 0),
                    "Profit Factor": report.get("Profit Factor", 0),
                    "Win Rate": report.get("Win Rate", 0),
                    "Total PnL": report.get("Total PnL", 0),
                    "Total Trades": report.get("Total Trades", 0),
                    "Wins": report.get("Wins", 0),
                    "Losses": report.get("Losses", 0)
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return df.sort_values(
            by=["AI Score", "Profit Factor", "Win Rate", "Total PnL"],
            ascending=[False, False, False, False]
        )