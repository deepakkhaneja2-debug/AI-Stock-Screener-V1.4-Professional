import math
import pandas as pd

from config import *


class RiskEngine:
    """Calculates entry, stop‑loss, targets, and position size."""

    def __init__(self):
        self.risk_percent = RISK_PER_TRADE
        self.entry_atr_buffer = ENTRY_ATR_BUFFER
        self.stop_atr_multiplier = STOP_ATR_MULTIPLIER
        self.target1_r = TARGET1_R
        self.target2_r = TARGET2_R
        self.target3_r = TARGET3_R

    def _safe_get(self, data: pd.DataFrame, column: str, default: float = 0.0) -> float:
        if data.empty or column not in data.columns:
            return default
        try:
            val = data[column].iloc[-1]
            return float(val) if pd.notna(val) else default
        except Exception:
            return default

    def trade_plan(self, data: pd.DataFrame, capital: float) -> dict:
        """Return a complete trade plan with entry, targets, stop, and quantity."""
        if data.empty:
            return {}

        required = ["Close", "ATR"]
        if not all(col in data.columns for col in required):
            return {}

        current_price = self._safe_get(data, "Close")
        atr = self._safe_get(data, "ATR")

        if atr <= 0 or current_price <= 0:
            return {}

        entry = round(current_price + atr * self.entry_atr_buffer, 2)
        stoploss = round(entry - atr * self.stop_atr_multiplier, 2)

        risk = entry - stoploss
        if risk <= 0:
            return {}

        target1 = round(entry + risk * self.target1_r, 2)
        target2 = round(entry + risk * self.target2_r, 2)
        target3 = round(entry + risk * self.target3_r, 2)

        qty = self._quantity(capital, entry, stoploss)
        rr = round((target2 - entry) / risk, 2) if risk > 0 else 0

        return {
            "CurrentPrice": round(current_price, 2),
            "Entry": entry,
            "StopLoss": stoploss,
            "Target1": target1,
            "Target2": target2,
            "Target3": target3,
            "RR": rr,
            "Quantity": qty,
            "RiskScore": self._risk_score(entry, stoploss),
            "RewardScore": self._reward_score(entry, target2),
        }

    def _quantity(self, capital: float, entry: float, sl: float) -> int:
        """Calculate position size based on risk."""
        if capital <= 0:
            return 0
        risk_amount = capital * self.risk_percent / 100
        risk_per_share = abs(entry - sl)
        if risk_per_share <= 0:
            return 0
        qty = math.floor(risk_amount / risk_per_share)
        return max(qty, 1)

    def _risk_score(self, entry: float, sl: float) -> int:
        risk = abs(entry - sl)
        if risk <= 1:
            return 100
        if risk <= 2:
            return 80
        if risk <= 3:
            return 60
        return 40

    def _reward_score(self, entry: float, target: float) -> int:
        reward = abs(target - entry)
        if reward >= 6:
            return 100
        if reward >= 4:
            return 80
        if reward >= 2:
            return 60
        return 40