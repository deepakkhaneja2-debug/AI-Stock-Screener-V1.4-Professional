import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

import config


class BacktestAnalyzer:
    """
    Advanced backtest analysis engine for AI Stock Scanner V1.4.
    Calculates comprehensive performance metrics.

    This is the SINGLE canonical source of derived statistics shown
    in the dashboard - it recomputes everything from the raw closed
    "Trades" list produced by BacktestEngine so there is only one
    place where Win Rate / Profit Factor / Expectancy / AI Score are
    defined (STEP 25: avoid inconsistent duplicate calculations).
    """

    def __init__(self):
        self.min_trades_for_ranking = getattr(config, "MIN_TRADES_FOR_RANKING", 5)

    def analyze(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a backtest report and return comprehensive metrics."""
        if not isinstance(report, dict):
            return self._empty_analysis()

        trades = report.get("Trades", [])
        if not trades or not isinstance(trades, list):
            empty = self._empty_analysis()
            empty["Debug"] = report.get("Debug", {}) if isinstance(report, dict) else {}
            return empty

        total_trades = len(trades)

        if total_trades == 0:
            return self._empty_analysis()

        # Win/Loss/Break-even counts
        wins = sum(1 for t in trades if t.get("Status") == "WIN")
        losses = sum(1 for t in trades if t.get("Status") == "LOSS")
        break_even = sum(1 for t in trades if t.get("Status") == "BREAK_EVEN")

        # Win/Loss rates (of ALL closed trades, break-even included in denominator
        # since it's still an outcome of the strategy)
        win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0
        loss_rate = round((losses / total_trades) * 100, 2) if total_trades > 0 else 0.0

        # P&L calculations - NetPnL is canonical (STEP 11/25), PnL is an alias
        pnls = [float(t.get("NetPnL", t.get("PnL", 0)) or 0) for t in trades]
        total_pnl = round(sum(pnls), 2)

        profits = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p < 0]

        avg_profit = round(sum(profits) / len(profits), 2) if profits else 0.0
        avg_loss = round(sum(losses_list) / len(losses_list), 2) if losses_list else 0.0

        # ---- Profit factor (STEP 12) ----
        gross_profit = sum(profits)
        gross_loss = abs(sum(losses_list))
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            # No losing trades at all - genuinely infinite, not an
            # arbitrary "999". The dashboard shows the infinity symbol
            # only for this exact case.
            profit_factor = float("inf")
        else:
            profit_factor = None

        # ---- Expectancy (STEP 13) ----
        # Break-even trades should not distort win/loss probability,
        # so the probabilities below are computed over DECISIVE
        # (win or loss) trades only.
        decisive = wins + losses
        win_prob = (wins / decisive) if decisive > 0 else 0.0
        loss_prob = (losses / decisive) if decisive > 0 else 0.0

        expectancy = round((win_prob * avg_profit) + (loss_prob * avg_loss), 2) if decisive > 0 else 0.0

        # R-based expectancy
        r_multiples_all = [float(t.get("RMultiple", 0) or 0) for t in trades]
        win_r = [r for r, t in zip(r_multiples_all, trades) if t.get("Status") == "WIN"]
        loss_r = [r for r, t in zip(r_multiples_all, trades) if t.get("Status") == "LOSS"]
        avg_win_r = round(sum(win_r) / len(win_r), 2) if win_r else 0.0
        avg_loss_r = round(sum(loss_r) / len(loss_r), 2) if loss_r else 0.0
        expectancy_r = round((win_prob * avg_win_r) + (loss_prob * avg_loss_r), 2) if decisive > 0 else 0.0

        # Best and worst trades
        best_trade = round(max(pnls), 2) if pnls else 0.0
        worst_trade = round(min(pnls), 2) if pnls else 0.0

        # Consecutive wins and losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for t in trades:
            status = t.get("Status")
            if status == "WIN":
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif status == "LOSS":
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            elif status == "BREAK_EVEN":
                current_wins = 0
                current_losses = 0

        # Average R multiple (all closed trades, including break-even at ~0R)
        avg_r_multiple = round(sum(r_multiples_all) / len(r_multiples_all), 2) if r_multiples_all else 0.0

        # Average holding days
        holding_days = [
            int(t.get("HoldingDays", 0))
            for t in trades
            if t.get("HoldingDays", 0) is not None and t.get("HoldingDays", 0) > 0
        ]
        avg_holding_days = round(sum(holding_days) / len(holding_days), 1) if holding_days else 0.0

        # Target hits
        target1_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET1")
        target2_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET2")
        target3_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET3")

        # ---- Max drawdown (STEP 14): starts at STARTING_CAPITAL, not zero ----
        starting_capital = getattr(config, "STARTING_CAPITAL", 100000)
        equity = starting_capital
        peak = starting_capital
        max_drawdown = 0.0

        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            drawdown = equity - peak
            max_drawdown = min(max_drawdown, drawdown)

        max_drawdown = round(max_drawdown, 2)
        max_drawdown_pct = round((max_drawdown / peak) * 100, 2) if peak else 0.0

        # Monthly P&L
        monthly_pnl = self._calculate_monthly_pnl(trades)

        # Equity curve (starts at starting capital, per STEP 14)
        equity_curve = self._calculate_equity_curve(trades, starting_capital)

        # Drawdown curve
        drawdown_curve = self._calculate_drawdown_curve(trades, starting_capital)

        # AI Score (0-100) + Data Quality (STEP 15)
        ai_score, data_quality = self._calculate_ai_score(
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            avg_r_multiple=avg_r_multiple,
            wins=wins,
            losses=losses
        )

        # Strategy status
        pf_for_compare = profit_factor if profit_factor is not None else 0.0
        if total_trades < 10:
            strategy_status = "INSUFFICIENT DATA"
        elif pf_for_compare > 1.5 and expectancy > 0 and win_rate > 50:
            strategy_status = "STRONG"
        elif pf_for_compare > 1.0 and expectancy > 0:
            strategy_status = "PROFITABLE"
        else:
            strategy_status = "WEAK"

        return {
            "Total Trades": total_trades,
            "Closed Trades": total_trades,
            "Open Trades": 0,
            "Wins": wins,
            "Losses": losses,
            "Break Even": break_even,
            "Win Rate": win_rate,
            "Loss Rate": loss_rate,
            "Total PnL": total_pnl,
            "Total Return %": round((total_pnl / starting_capital) * 100, 2) if starting_capital else 0.0,
            "Average Profit": avg_profit,
            "Average Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Expectancy": expectancy,
            "Expectancy R": expectancy_r,
            "Avg Win R": avg_win_r,
            "Avg Loss R": avg_loss_r,
            "Max Drawdown": max_drawdown,
            "Max Drawdown %": max_drawdown_pct,
            "Best Trade": best_trade,
            "Worst Trade": worst_trade,
            "Consecutive Wins": max_consecutive_wins,
            "Consecutive Losses": max_consecutive_losses,
            "Average Holding Days": avg_holding_days,
            "Average R Multiple": avg_r_multiple,
            "Monthly PnL": monthly_pnl,
            "Equity Curve": equity_curve,
            "Drawdown Curve": drawdown_curve,
            "AI Score": ai_score,
            "Data Quality": data_quality,
            "Strategy Status": strategy_status,
            "Target1 Wins": target1_wins,
            "Target2 Wins": target2_wins,
            "Target3 Wins": target3_wins,
            "Trades": trades,
            "Debug": report.get("Debug", {}),
        }

    def _calculate_monthly_pnl(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate monthly P&L from trade history."""
        monthly = {}
        for t in trades:
            exit_date = t.get("ExitDate")
            if exit_date is not None:
                try:
                    if hasattr(exit_date, "strftime"):
                        month_key = exit_date.strftime("%Y-%m")
                    else:
                        try:
                            date_obj = pd.to_datetime(exit_date)
                            month_key = date_obj.strftime("%Y-%m")
                        except Exception:
                            continue
                    pnl = float(t.get("NetPnL", t.get("PnL", 0)) or 0)
                    monthly[month_key] = monthly.get(month_key, 0) + pnl
                except Exception:
                    continue
        return monthly

    def _calculate_equity_curve(self, trades: List[Dict], starting_capital: float = 0.0) -> List[Dict]:
        """Calculate equity curve from trade history, starting at starting_capital (STEP 14)."""
        if not trades:
            return []

        sorted_trades = sorted(
            trades,
            key=lambda x: x.get("ExitDate", pd.Timestamp.min)
        )

        equity = starting_capital
        curve = [{"Date": "Start", "Equity": round(equity, 2)}]
        for t in sorted_trades:
            exit_date = t.get("ExitDate")
            if exit_date is not None:
                try:
                    if hasattr(exit_date, "strftime"):
                        date_str = exit_date.strftime("%Y-%m-%d")
                    else:
                        try:
                            date_obj = pd.to_datetime(exit_date)
                            date_str = date_obj.strftime("%Y-%m-%d")
                        except Exception:
                            continue
                    pnl = float(t.get("NetPnL", t.get("PnL", 0)) or 0)
                    equity += pnl
                    curve.append({
                        "Date": date_str,
                        "Equity": round(equity, 2)
                    })
                except Exception:
                    continue
        return curve

    def _calculate_drawdown_curve(self, trades: List[Dict], starting_capital: float = 0.0) -> List[Dict]:
        """Calculate drawdown curve from trade history, starting at starting_capital (STEP 14)."""
        if not trades:
            return []

        sorted_trades = sorted(
            trades,
            key=lambda x: x.get("ExitDate", pd.Timestamp.min)
        )

        equity = starting_capital
        peak = starting_capital
        curve = []
        for t in sorted_trades:
            exit_date = t.get("ExitDate")
            if exit_date is not None:
                try:
                    if hasattr(exit_date, "strftime"):
                        date_str = exit_date.strftime("%Y-%m-%d")
                    else:
                        try:
                            date_obj = pd.to_datetime(exit_date)
                            date_str = date_obj.strftime("%Y-%m-%d")
                        except Exception:
                            continue
                    pnl = float(t.get("NetPnL", t.get("PnL", 0)) or 0)
                    equity += pnl
                    peak = max(peak, equity)
                    drawdown = round(equity - peak, 2)
                    curve.append({
                        "Date": date_str,
                        "Drawdown": drawdown
                    })
                except Exception:
                    continue
        return curve

    def _calculate_ai_score(
        self,
        win_rate: float,
        profit_factor: Optional[float],
        expectancy: float,
        max_drawdown: float,
        total_trades: int,
        avg_r_multiple: float,
        wins: int,
        losses: int
    ):
        """
        Calculate AI Score (0-100) based on multiple performance metrics
        PLUS a Data Quality label (STEP 15). A strategy is never scored
        highly just because it has a high win rate on a tiny sample -
        the sample-size adjustment below caps that.
        """
        if total_trades == 0:
            return 0, "NO TRADES"

        pf = profit_factor if profit_factor is not None else 0.0
        pf_is_infinite = profit_factor == float("inf")

        score = 50.0

        # Win rate component
        if win_rate >= 60:
            score += 10
        elif win_rate >= 50:
            score += 5
        elif win_rate >= 40:
            score -= 5
        else:
            score -= 10

        # Profit factor component
        if pf_is_infinite or pf >= 2.0:
            score += 15
        elif pf >= 1.5:
            score += 10
        elif pf >= 1.0:
            score += 5
        else:
            score -= 10

        # Expectancy component
        if expectancy >= 2.0:
            score += 10
        elif expectancy >= 1.0:
            score += 5
        elif expectancy >= 0.0:
            score += 2
        else:
            score -= 5

        # Drawdown penalty
        if max_drawdown <= -100:
            score -= 20
        elif max_drawdown <= -50:
            score -= 15
        elif max_drawdown <= -20:
            score -= 10
        elif max_drawdown <= -10:
            score -= 5

        # R multiple bonus
        if avg_r_multiple >= 2.0:
            score += 10
        elif avg_r_multiple >= 1.5:
            score += 5
        elif avg_r_multiple >= 1.0:
            score += 2

        # ---- Sample-size adjustment (STEP 15) ----
        # Never artificially boost small-sample scores; only ever
        # dampen them toward a neutral value as sample size shrinks.
        if total_trades < 5:
            score = min(score, 40) * 0.4
            data_quality = "INSUFFICIENT DATA"
        elif total_trades < 10:
            score *= 0.6
            data_quality = "LOW SAMPLE"
        elif total_trades < 30:
            score *= 0.85
            data_quality = "MODERATE SAMPLE"
        elif total_trades < 50:
            data_quality = "GOOD SAMPLE"
        else:
            score *= 1.05
            data_quality = "STRONG SAMPLE"

        return max(0, min(100, int(round(score)))), data_quality

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis dictionary."""
        return {
            "Total Trades": 0,
            "Closed Trades": 0,
            "Open Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Break Even": 0,
            "Win Rate": 0.0,
            "Loss Rate": 0.0,
            "Total PnL": 0.0,
            "Total Return %": 0.0,
            "Average Profit": 0.0,
            "Average Loss": 0.0,
            "Profit Factor": None,
            "Expectancy": 0.0,
            "Expectancy R": 0.0,
            "Avg Win R": 0.0,
            "Avg Loss R": 0.0,
            "Max Drawdown": 0.0,
            "Max Drawdown %": 0.0,
            "Best Trade": 0.0,
            "Worst Trade": 0.0,
            "Consecutive Wins": 0,
            "Consecutive Losses": 0,
            "Average Holding Days": 0.0,
            "Average R Multiple": 0.0,
            "Monthly PnL": {},
            "Equity Curve": [],
            "Drawdown Curve": [],
            "AI Score": 0,
            "Data Quality": "NO TRADES",
            "Strategy Status": "NO TRADES",
            "Target1 Wins": 0,
            "Target2 Wins": 0,
            "Target3 Wins": 0,
            "Trades": [],
            "Debug": {},
        }
