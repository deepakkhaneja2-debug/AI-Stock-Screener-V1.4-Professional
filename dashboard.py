import math
import pandas as pd
from typing import Dict, Any, Optional

import config


def _pf_sort_key(value) -> float:
    """Make Profit Factor sortable even when it's None or float('inf')."""
    if value is None:
        return -1.0
    if isinstance(value, float) and math.isinf(value):
        return 1e12
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def format_profit_factor(value) -> str:
    """Display helper: None -> N/A, inf -> the infinity symbol, else 2dp (STEP 12)."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "\u221e"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


class DashboardEngine:
    """Generates dashboard views and statistics (STEP 16 / 17)."""

    def __init__(self):
        self.min_trades_for_ranking = getattr(config, "MIN_TRADES_FOR_RANKING", 5)

    # ------------------------------------------------------------
    # OVERALL STATS (App.py "Overall Backtest Dashboard")
    # ------------------------------------------------------------
    def overall_stats(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall statistics from backtest reports.
        Each report is a dictionary containing per-symbol backtest results.
        """
        empty = {
            "Total Trades": 0,
            "Closed Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "BreakEven": 0,
            "Win Rate": 0.0,
            "Total PnL": 0.0,
            "Return %": 0.0,
            "Profit Factor": None,
            "Expectancy": 0.0,
            "Average R": 0.0,
            "Max Drawdown": 0.0,
            "AI Score": 0,
        }

        if not reports:
            return empty

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
            return empty

        total = len(all_trades)
        wins = sum(1 for t in all_trades if t.get("Status") == "WIN")
        losses = sum(1 for t in all_trades if t.get("Status") == "LOSS")
        break_even = sum(1 for t in all_trades if t.get("Status") == "BREAK_EVEN")
        closed = wins + losses + break_even

        win_rate = round((wins / closed) * 100, 2) if closed > 0 else 0.0

        pnls = [float(t.get("NetPnL", t.get("PnL", 0)) or 0) for t in all_trades]
        total_pnl = round(sum(pnls), 2)

        profits = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p < 0]

        gross_profit = sum(profits)
        gross_loss = abs(sum(losses_list))

        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = None

        decisive = wins + losses
        win_prob = wins / decisive if decisive > 0 else 0.0
        loss_prob = losses / decisive if decisive > 0 else 0.0
        avg_win = round(sum(profits) / len(profits), 2) if profits else 0.0
        avg_loss = round(sum(losses_list) / len(losses_list), 2) if losses_list else 0.0
        expectancy = round((win_prob * avg_win) + (loss_prob * avg_loss), 2) if decisive > 0 else 0.0

        r_multiples = [float(t.get("RMultiple", 0) or 0) for t in all_trades]
        avg_r = round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0.0

        # Sort chronologically by ExitDate before computing the portfolio
        # equity/drawdown curve - iterating in symbol-append order (the
        # previous behaviour) produced an incorrect drawdown because
        # trades from different symbols were interleaved out of order.
        all_trades_sorted = sorted(
            all_trades,
            key=lambda t: t.get("ExitDate") or pd.Timestamp.min
        )
        pnls_sorted = [float(t.get("NetPnL", t.get("PnL", 0)) or 0) for t in all_trades_sorted]

        starting_capital = getattr(config, "STARTING_CAPITAL", 100000)
        equity = starting_capital
        peak = starting_capital
        max_dd = 0.0
        for pnl in pnls_sorted:
            equity += pnl
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        avg_ai_score = round(sum(ai_scores) / len(ai_scores), 0) if ai_scores else 0

        return {
            "Total Trades": total,
            "Closed Trades": closed,
            "Wins": wins,
            "Losses": losses,
            "BreakEven": break_even,
            "Win Rate": win_rate,
            "Total PnL": total_pnl,
            "Return %": round((total_pnl / starting_capital) * 100, 2) if starting_capital else 0.0,
            "Profit Factor": profit_factor,
            "Expectancy": expectancy,
            "Average R": avg_r,
            "Max Drawdown": round(max_dd, 2),
            "AI Score": avg_ai_score,
        }

    # ------------------------------------------------------------
    # HIGHLIGHTS (STEP 17: Strongest Stock, Best AI Score, etc.)
    # ------------------------------------------------------------
    def highlights(self, reports: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Return the standout symbol for each headline metric. Only
        symbols with at least MIN_TRADES_FOR_RANKING closed trades are
        considered "reliable" for the AI-Score / most-reliable picks,
        so a fluke 2-trade sample can't claim the top spot.
        """
        result = {
            "Strongest Stock": None,
            "Best AI Score": None,
            "Best PnL": None,
            "Highest Win Rate": None,
            "Most Reliable Stock": None,
        }

        if not reports:
            return result

        reliable = {
            s: r for s, r in reports.items()
            if isinstance(r, dict) and r.get("Total Trades", 0) >= self.min_trades_for_ranking
        }
        pool = reliable if reliable else {
            s: r for s, r in reports.items() if isinstance(r, dict) and r.get("Total Trades", 0) > 0
        }

        if not pool:
            return result

        def _best(metric, key=None):
            items = [(s, r.get(metric, 0)) for s, r in pool.items()]
            items = [(s, v) for s, v in items if v is not None]
            if not items:
                return None
            return max(items, key=lambda x: (key or (lambda v: v))(x[1]))[0]

        result["Best AI Score"] = _best("AI Score")
        result["Best PnL"] = _best("Total PnL")
        result["Highest Win Rate"] = _best("Win Rate")
        result["Most Reliable Stock"] = _best("AI Score")  # reliable pool already filters by sample size
        # "Strongest Stock" = highest AI Score among symbols with a positive Profit Factor
        pf_positive = {
            s: r for s, r in pool.items()
            if (r.get("Profit Factor") is None or (isinstance(r.get("Profit Factor"), float) and math.isinf(r.get("Profit Factor")))
                or r.get("Profit Factor", 0) >= 1.0)
        }
        base = pf_positive if pf_positive else pool
        best_symbol = None
        best_score = -1
        for s, r in base.items():
            score = r.get("AI Score", 0) or 0
            if score > best_score:
                best_score = score
                best_symbol = s
        result["Strongest Stock"] = best_symbol

        return result

    # ------------------------------------------------------------
    # RANKING TABLE (STEP 16)
    # ------------------------------------------------------------
    def ranking_table(self, reports: Dict[str, Any]):
        """
        Build the stock ranking table.

        Returns a dict with two DataFrames:
          "ranked"     - symbols with >= MIN_TRADES_FOR_RANKING closed
                         trades, sorted AI Score > Profit Factor >
                         Win Rate > Total PnL.
          "low_sample" - everything else, shown separately so a
                         2-trade fluke never outranks a proven strategy.
        """
        empty = pd.DataFrame()
        if not reports:
            return {"ranked": empty, "low_sample": empty}

        rows = []
        for symbol, report in reports.items():
            if not isinstance(report, dict):
                continue
            total_trades = report.get("Total Trades", 0)
            rows.append({
                "Symbol": symbol,
                "Signal": report.get("Signal", "-"),
                "Total Trades": total_trades,
                "Win Rate": report.get("Win Rate", 0),
                "Total PnL": report.get("Total PnL", 0),
                "Profit Factor": report.get("Profit Factor", None),
                "Profit Factor Display": format_profit_factor(report.get("Profit Factor", None)),
                "Expectancy": report.get("Expectancy", 0),
                "Average R": report.get("Average R Multiple", report.get("Average R", 0)),
                "Max Drawdown": report.get("Max Drawdown", 0),
                "AI Score": report.get("AI Score", 0),
                "Data Quality": report.get("Data Quality", "NO TRADES"),
                "_pf_sort": _pf_sort_key(report.get("Profit Factor", None)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return {"ranked": empty, "low_sample": empty}

        ranked = df[df["Total Trades"] >= self.min_trades_for_ranking].copy()
        low_sample = df[df["Total Trades"] < self.min_trades_for_ranking].copy()

        if not ranked.empty:
            ranked = ranked.sort_values(
                by=["AI Score", "_pf_sort", "Win Rate", "Total PnL"],
                ascending=[False, False, False, False]
            )
        if not low_sample.empty:
            low_sample = low_sample.sort_values(
                by=["AI Score", "_pf_sort"],
                ascending=[False, False]
            )

        display_cols = [
            "Symbol", "Signal", "Total Trades", "Win Rate", "Total PnL",
            "Profit Factor Display", "Expectancy", "Average R",
            "Max Drawdown", "AI Score", "Data Quality"
        ]
        rename = {"Profit Factor Display": "Profit Factor"}

        ranked_out = ranked[display_cols].rename(columns=rename) if not ranked.empty else empty
        low_sample_out = low_sample[display_cols].rename(columns=rename) if not low_sample.empty else empty

        return {"ranked": ranked_out, "low_sample": low_sample_out}
