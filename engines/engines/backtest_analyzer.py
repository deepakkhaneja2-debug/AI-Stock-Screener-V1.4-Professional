import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class BacktestAnalyzer:
    """
    Advanced backtest analysis engine for AI Stock Scanner V1.4.
    """

    def __init__(self):
        self.min_trades_for_ranking = 5

    def analyze(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a backtest report and return comprehensive metrics."""
        if not isinstance(report, dict):
            return self._empty_analysis()

        trades = report.get("Trades", [])
        if not trades or not isinstance(trades, list):
            return self._empty_analysis()

        total_trades = len(trades)

        if total_trades == 0:
            return self._empty_analysis()

        wins = sum(1 for t in trades if t.get("Status") == "WIN")
        losses = sum(1 for t in trades if t.get("Status") == "LOSS")
        break_even = sum(1 for t in trades if t.get("Status") == "BREAK_EVEN")

        win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0
        loss_rate = round((losses / total_trades) * 100, 2) if total_trades > 0 else 0.0

        pnls = [float(t.get("PnL", 0)) for t in trades]
        total_pnl = round(sum(pnls), 2)

        profits = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p < 0]

        avg_profit = round(sum(profits) / len(profits), 2) if profits else 0.0
        avg_loss = round(sum(losses_list) / len(losses_list), 2) if losses_list else 0.0

        gross_profit = sum(profits)
        gross_loss = abs(sum(losses_list))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

        expectancy = round(
            (wins / total_trades * avg_profit) + (losses / total_trades * avg_loss),
            2
        ) if total_trades > 0 else 0.0

        best_trade = round(max(pnls), 2) if pnls else 0.0
        worst_trade = round(min(pnls), 2) if pnls else 0.0

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

        r_multiples = [float(t.get("RMultiple", 0)) for t in trades if t.get("RMultiple", 0) != 0]
        avg_r_multiple = round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0.0

        holding_days = [
            int(t.get("HoldingDays", 0))
            for t in trades
            if t.get("HoldingDays", 0) is not None and t.get("HoldingDays", 0) > 0
        ]
        avg_holding_days = round(sum(holding_days) / len(holding_days), 1) if holding_days else 0.0

        target1_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET1")
        target2_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET2")
        target3_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET3")

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            drawdown = equity - peak
            max_drawdown = min(max_drawdown, drawdown)

        max_drawdown = round(max_drawdown, 2)

        monthly_pnl = self._calculate_monthly_pnl(trades)
        equity_curve = self._calculate_equity_curve(trades)
        drawdown_curve = self._calculate_drawdown_curve(trades)

        ai_score = self._calculate_ai_score(
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            avg_r_multiple=avg_r_multiple,
            wins=wins,
            losses=losses
        )

        if total_trades < 10:
            strategy_status = "INSUFFICIENT DATA"
        elif profit_factor > 1.5 and expectancy > 0 and win_rate > 50:
            strategy_status = "STRONG"
        elif profit_factor > 1.0 and expectancy > 0:
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
            "Average Profit": avg_profit,
            "Average Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Expectancy": expectancy,
            "Max Drawdown": max_drawdown,
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
            "Strategy Status": strategy_status,
            "Target1 Wins": target1_wins,
            "Target2 Wins": target2_wins,
            "Target3 Wins": target3_wins,
            "Trades": trades
        }

    def _calculate_monthly_pnl(self, trades: List[Dict]) -> Dict[str, float]:
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
                    pnl = float(t.get("PnL", 0))
                    monthly[month_key] = monthly.get(month_key, 0) + pnl
                except Exception:
                    continue
        return monthly

    def _calculate_equity_curve(self, trades: List[Dict]) -> List[Dict]:
        if not trades:
            return []

        sorted_trades = sorted(
            trades,
            key=lambda x: x.get("ExitDate", pd.Timestamp.min)
        )

        equity = 0.0
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
                    pnl = float(t.get("PnL", 0))
                    equity += pnl
                    curve.append({
                        "Date": date_str,
                        "Equity": round(equity, 2)
                    })
                except Exception:
                    continue
        return curve

    def _calculate_drawdown_curve(self, trades: List[Dict]) -> List[Dict]:
        if not trades:
            return []

        sorted_trades = sorted(
            trades,
            key=lambda x: x.get("ExitDate", pd.Timestamp.min)
        )

        equity = 0.0
        peak = 0.0
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
                    pnl = float(t.get("PnL", 0))
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
        profit_factor: float,
        expectancy: float,
        max_drawdown: float,
        total_trades: int,
        avg_r_multiple: float,
        wins: int,
        losses: int
    ) -> int:
        if total_trades == 0:
            return 0

        score = 50.0

        if win_rate >= 60:
            score += 10
        elif win_rate >= 50:
            score += 5
        elif win_rate >= 40:
            score -= 5
        else:
            score -= 10

        if profit_factor >= 2.0:
            score += 15
        elif profit_factor >= 1.5:
            score += 10
        elif profit_factor >= 1.0:
            score += 5
        else:
            score -= 10

        if expectancy >= 2.0:
            score += 10
        elif expectancy >= 1.0:
            score += 5
        elif expectancy >= 0.0:
            score += 2
        else:
            score -= 5

        if max_drawdown <= -100:
            score -= 20
        elif max_drawdown <= -50:
            score -= 15
        elif max_drawdown <= -20:
            score -= 10
        elif max_drawdown <= -10:
            score -= 5

        if avg_r_multiple >= 2.0:
            score += 10
        elif avg_r_multiple >= 1.5:
            score += 5
        elif avg_r_multiple >= 1.0:
            score += 2

        if total_trades < 10:
            score *= 0.5
        elif total_trades >= 50:
            score *= 1.1

        return max(0, min(100, int(round(score))))

    def _empty_analysis(self) -> Dict[str, Any]:
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
            "Average Profit": 0.0,
            "Average Loss": 0.0,
            "Profit Factor": 0.0,
            "Expectancy": 0.0,
            "Max Drawdown": 0.0,
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
            "Strategy Status": "NO TRADES",
            "Target1 Wins": 0,
            "Target2 Wins": 0,
            "Target3 Wins": 0,
            "Trades": []
        }