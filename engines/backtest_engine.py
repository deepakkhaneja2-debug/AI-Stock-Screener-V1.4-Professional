import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

from config import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BacktestEngine:
    """
    Enhanced backtest engine with proper trade management.
    """

    def __init__(self):
        self.lookahead_days = BACKTEST_LOOKAHEAD
        self.entry_atr_buffer = ENTRY_ATR_BUFFER
        self.stop_atr_multiplier = STOP_ATR_MULTIPLIER
        self.target1_r = TARGET1_R
        self.target2_r = TARGET2_R
        self.target3_r = TARGET3_R
        self.use_break_even = BREAK_EVEN_AT_TARGET1
        self.trailing_stop_atr = TRAILING_STOP_ATR
        self.min_trades_for_ranking = MIN_TRADES_FOR_RANKING
        self.risk_per_trade = 1.0
        self.initial_capital = 100000

    def run(self, data: pd.DataFrame) -> dict:
        """Run backtest with full trade management."""
        if data is None or data.empty:
            logger.warning("No data provided for backtest")
            return self._summary([])

        data = data.copy()
        required = ["Close", "High", "Low", "EMA20", "EMA50", "RSI", "MACD", 
                    "MACD_SIGNAL", "ATR", "VWAP", "SMA50", "SMA200"]

        if not all(col in data.columns for col in required):
            logger.warning("Missing required columns for backtest")
            return self._summary([])

        data = data.dropna(subset=required)
        if len(data) < 61:
            logger.warning("Insufficient data for backtest")
            return self._summary([])

        data = data.sort_index()
        next_available_index = 60
        results = []
        quantity = 1

        for i in range(60, len(data) - 1):
            if i < next_available_index:
                continue

            row = data.iloc[i]

            try:
                ema20 = float(row["EMA20"])
                ema50 = float(row["EMA50"])
                rsi = float(row["RSI"])
                macd = float(row["MACD"])
                macd_signal = float(row["MACD_SIGNAL"])
                close = float(row["Close"])
                atr = float(row["ATR"])
                vwap = float(row["VWAP"])
                sma50 = float(row["SMA50"])
                sma200 = float(row["SMA200"])
            except (TypeError, ValueError) as e:
                logger.debug(f"Error reading indicators at {i}: {e}")
                continue

            if atr <= 0 or close <= 0:
                continue

            # STRICT BUY SETUP
            buy_setup = (
                ema20 > ema50 * 1.01 and
                macd > macd_signal and
                macd > 0.5 and
                macd_signal > 0 and
                58 < rsi < 75 and
                close > vwap * 1.005 and
                close > ema20 and
                close > sma50 and
                close > sma200
            )

            # Breakout confirmation
            if "BREAKOUT_UP" in data.columns:
                try:
                    breakout = bool(data["BREAKOUT_UP"].iloc[i])
                    buy_setup = buy_setup and breakout
                except:
                    pass

            if not buy_setup:
                continue

            # Calculate entry and stop loss
            entry = round(close + atr * self.entry_atr_buffer, 2)
            stoploss = round(entry - atr * self.stop_atr_multiplier, 2)
            risk = round(entry - stoploss, 2)

            if risk <= 0:
                continue

            target1 = round(entry + risk * self.target1_r, 2)
            target2 = round(entry + risk * self.target2_r, 2)
            target3 = round(entry + risk * self.target3_r, 2)

            # Find entry trigger
            entry_index = None
            for j in range(i + 1, min(i + 10, len(data))):
                try:
                    if float(data.iloc[j]["High"]) >= entry:
                        entry_index = j
                        break
                except (TypeError, ValueError):
                    continue

            if entry_index is None:
                continue

            # Initialize trade
            trade = {
                "Date": data.iloc[entry_index].name,
                "SignalDate": row.name,
                "Entry": entry,
                "StopLoss": stoploss,
                "Risk": risk,
                "Target1": target1,
                "Target2": target2,
                "Target3": target3,
                "Expiry": min(entry_index + self.lookahead_days, len(data)),
                "Status": "OPEN",
                "EntryIndex": entry_index,
                "CurrentStop": stoploss,
                "HighestPrice": entry,
                "LowestPrice": entry,
                "TargetHit": None,
                "ExitPrice": None,
                "ExitDate": None,
                "ExitReason": None,
                "HoldingDays": 0,
                "PnL": 0.0,
                "PnLPercent": 0.0,
                "RMultiple": 0.0,
                "TradeReason": "STRICT BUY SETUP",
                "RR": round((target2 - entry) / risk, 2) if risk > 0 else 0,
                "Quantity": quantity
            }

            # --- TRADE MANAGEMENT ---
            trade_closed = False
            for j in range(entry_index, trade["Expiry"]):
                candle = data.iloc[j]
                try:
                    low = float(candle["Low"])
                    high = float(candle["High"])
                    candle_close = float(candle["Close"])
                    atr = float(candle["ATR"]) if "ATR" in data.columns else trade["Risk"] / 1.5
                except (TypeError, ValueError):
                    continue

                trade["HoldingDays"] = j - trade["EntryIndex"] + 1
                trade["HighestPrice"] = max(trade["HighestPrice"], high)
                trade["LowestPrice"] = min(trade["LowestPrice"], low)

                # Update trailing stop
                profit = high - trade["Entry"]
                if profit >= trade["Risk"] * 1.5:
                    trade["CurrentStop"] = max(
                        trade["CurrentStop"],
                        high - atr * self.trailing_stop_atr
                    )

                # Check targets
                if high >= trade["Target3"]:
                    self._close_trade(trade, j, "TARGET3")
                    trade_closed = True
                    break
                elif high >= trade["Target2"]:
                    self._close_trade(trade, j, "TARGET2")
                    trade_closed = True
                    break
                elif high >= trade["Target1"]:
                    if self.use_break_even:
                        trade["CurrentStop"] = trade["Entry"]
                    else:
                        self._close_trade(trade, j, "TARGET1")
                        trade_closed = True
                        break

                # Check stop loss
                if low <= trade["CurrentStop"]:
                    self._close_trade(trade, j, "STOP_LOSS")
                    trade_closed = True
                    break

            if not trade_closed and trade["Status"] == "OPEN":
                last_idx = min(trade["Expiry"] - 1, len(data) - 1)
                self._close_trade(trade, last_idx, "TIME_EXIT")

            if trade["Status"] != "OPEN":
                results.append(trade)

            next_available_index = max(entry_index + 1, trade["Expiry"])

        return self._summary(results)

    def _close_trade(self, trade: dict, exit_idx: int, reason: str) -> None:
        """Close a trade and calculate P&L."""
        if trade["Status"] != "OPEN":
            return

        if reason == "TARGET3":
            exit_price = trade["Target3"]
        elif reason == "TARGET2":
            exit_price = trade["Target2"]
        elif reason == "TARGET1":
            exit_price = trade["Target1"]
        elif reason == "STOP_LOSS":
            exit_price = trade["CurrentStop"]
        elif reason == "BREAK_EVEN":
            exit_price = trade["Entry"]
        else:
            exit_price = trade["Entry"]

        pnl = round(exit_price - trade["Entry"], 2)
        pnl_percent = round((pnl / trade["Entry"]) * 100, 2) if trade["Entry"] > 0 else 0
        r_multiple = round(pnl / trade["Risk"], 2) if trade["Risk"] > 0 else 0

        if pnl > 0:
            status = "WIN"
        elif pnl < 0:
            status = "LOSS"
        else:
            status = "BREAK_EVEN"

        trade["Status"] = status
        trade["ExitPrice"] = exit_price
        trade["ExitDate"] = trade["Date"] + pd.Timedelta(days=trade["HoldingDays"])
        trade["ExitReason"] = reason
        trade["PnL"] = pnl
        trade["PnLPercent"] = pnl_percent
        trade["RMultiple"] = r_multiple

        if reason in ["TARGET1", "TARGET2", "TARGET3"]:
            trade["TargetHit"] = reason

    def _summary(self, trades: list) -> dict:
        """Generate comprehensive backtest summary."""
        if not trades:
            return self._empty_summary()

        wins = sum(1 for t in trades if t["Status"] == "WIN")
        losses = sum(1 for t in trades if t["Status"] == "LOSS")
        break_even = sum(1 for t in trades if t["Status"] == "BREAK_EVEN")
        closed = wins + losses + break_even

        win_rate = round((wins / closed) * 100, 2) if closed > 0 else 0
        total_pnl = round(sum(t["PnL"] for t in trades), 2)

        profits = [t["PnL"] for t in trades if t["Status"] == "WIN"]
        losses_list = [t["PnL"] for t in trades if t["Status"] == "LOSS"]

        avg_profit = round(sum(profits) / len(profits), 2) if profits else 0
        avg_loss = round(sum(losses_list) / len(losses_list), 2) if losses_list else 0

        gross_profit = sum(profits)
        gross_loss = abs(sum(losses_list))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

        expectancy = round((wins/closed * avg_profit) + (losses/closed * avg_loss), 2) if closed > 0 else 0

        r_values = [t["RMultiple"] for t in trades if t["Status"] in ["WIN", "LOSS"]]
        avg_r = round(sum(r_values) / len(r_values), 2) if r_values else 0

        hold_days = [t["HoldingDays"] for t in trades]
        avg_hold_days = round(sum(hold_days) / len(hold_days), 1) if hold_days else 0

        equity = 0
        peak = 0
        max_dd = 0
        for t in trades:
            equity += t["PnL"]
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
        max_drawdown = round(max_dd, 2)

        target1_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET1")
        target2_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET2")
        target3_wins = sum(1 for t in trades if t.get("TargetHit") == "TARGET3")

        data_quality = "NO CLOSED TRADES" if closed == 0 else \
                      "LOW SAMPLE" if closed < self.min_trades_for_ranking else "SUFFICIENT SAMPLE"

        return {
            "Total Trades": len(trades),
            "Wins": wins,
            "Losses": losses,
            "BreakEven": break_even,
            "Open": 0,
            "Closed Trades": closed,
            "Win Rate": win_rate,
            "Total PnL": total_pnl,
            "Average Profit": avg_profit,
            "Average Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Expectancy": expectancy,
            "Average R": avg_r,
            "Average Holding Days": avg_hold_days,
            "Max Drawdown": max_drawdown,
            "Target1 Wins": target1_wins,
            "Target2 Wins": target2_wins,
            "Target3 Wins": target3_wins,
            "Data Quality": data_quality,
            "Trades": trades
        }

    def _empty_summary(self) -> dict:
        return {
            "Total Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "BreakEven": 0,
            "Open": 0,
            "Closed Trades": 0,
            "Win Rate": 0,
            "Total PnL": 0,
            "Average Profit": 0,
            "Average Loss": 0,
            "Profit Factor": 0,
            "Expectancy": 0,
            "Average R": 0,
            "Average Holding Days": 0,
            "Max Drawdown": 0,
            "Target1 Wins": 0,
            "Target2 Wins": 0,
            "Target3 Wins": 0,
            "Data Quality": "NO TRADES",
            "Trades": []
        }