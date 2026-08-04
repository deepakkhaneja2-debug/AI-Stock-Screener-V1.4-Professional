import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatternEngine:
    """Detects candlestick patterns."""

    def bullish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        if data.empty or len(data) < 2:
            return pd.Series(dtype=bool)
        prev = data.shift(1)
        return (
            (prev["Close"] < prev["Open"]) &
            (data["Close"] > data["Open"]) &
            (data["Open"] < prev["Close"]) &
            (data["Close"] > prev["Open"])
        )

    def bearish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        if data.empty or len(data) < 2:
            return pd.Series(dtype=bool)
        prev = data.shift(1)
        return (
            (prev["Close"] > prev["Open"]) &
            (data["Close"] < data["Open"]) &
            (data["Open"] > prev["Close"]) &
            (data["Close"] < prev["Open"])
        )

    def hammer(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=bool)
        body = abs(data["Close"] - data["Open"])
        lower = data[["Open", "Close"]].min(axis=1) - data["Low"]
        upper = data["High"] - data[["Open", "Close"]].max(axis=1)
        return (lower > body * 2) & (upper < body)

    def shooting_star(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=bool)
        body = abs(data["Close"] - data["Open"])
        upper = data["High"] - data[["Open", "Close"]].max(axis=1)
        lower = data[["Open", "Close"]].min(axis=1) - data["Low"]
        return (upper > body * 2) & (lower < body)

    def doji(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=bool)
        body = abs(data["Close"] - data["Open"])
        candle = data["High"] - data["Low"]
        return body <= candle * 0.1

    def morning_star(self, data: pd.DataFrame) -> pd.Series:
        if data.empty or len(data) < 3:
            return pd.Series(dtype=bool)
        prev2 = data.shift(2)
        prev1 = data.shift(1)
        return (
            (prev2["Close"] < prev2["Open"]) &
            (abs(prev1["Close"] - prev1["Open"]) <
             abs(prev2["Close"] - prev2["Open"]) * 0.3) &
            (data["Close"] > data["Open"]) &
            (data["Close"] > (prev2["Open"] + prev2["Close"]) / 2)
        )

    def evening_star(self, data: pd.DataFrame) -> pd.Series:
        if data.empty or len(data) < 3:
            return pd.Series(dtype=bool)
        prev2 = data.shift(2)
        prev1 = data.shift(1)
        return (
            (prev2["Close"] > prev2["Open"]) &
            (abs(prev1["Close"] - prev1["Open"]) <
             abs(prev2["Close"] - prev2["Open"]) * 0.3) &
            (data["Close"] < data["Open"]) &
            (data["Close"] < (prev2["Open"] + prev2["Close"]) / 2)
        )

    def inside_bar(self, data: pd.DataFrame) -> pd.Series:
        if data.empty or len(data) < 2:
            return pd.Series(dtype=bool)
        prev = data.shift(1)
        return (data["High"] < prev["High"]) & (data["Low"] > prev["Low"])

    def consolidation(self, data: pd.DataFrame, bars: int = 10) -> pd.Series:
        if data.empty or len(data) < bars:
            return pd.Series(dtype=bool)
        high = data["High"].rolling(bars, min_periods=bars).max()
        low = data["Low"].rolling(bars, min_periods=bars).min()
        result = ((high - low) / low) < 0.03
        return result.fillna(False)

    def breakout(self, data: pd.DataFrame, bars: int = 20):
        if data.empty or len(data) < bars + 1:
            false_series = pd.Series(dtype=bool)
            return false_series, false_series
        resistance = data["High"].rolling(bars, min_periods=bars).max().shift(1)
        support = data["Low"].rolling(bars, min_periods=bars).min().shift(1)
        up = (data["Close"] > resistance).fillna(False)
        down = (data["Close"] < support).fillna(False)
        return up, down

    def fake_breakout(self, data: pd.DataFrame, bars: int = 20):
        if data.empty or len(data) < bars + 1:
            false_series = pd.Series(dtype=bool)
            return false_series, false_series
        resistance = data["High"].rolling(bars, min_periods=bars).max().shift(1)
        support = data["Low"].rolling(bars, min_periods=bars).min().shift(1)
        up = ((data["High"] > resistance) & (data["Close"] < resistance)).fillna(False)
        down = ((data["Low"] < support) & (data["Close"] > support)).fillna(False)
        return up, down

    def pattern_score(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0

        score = 0
        try:
            if self.bullish_engulfing(data).iloc[-1]:
                score += 25
            if self.hammer(data).iloc[-1]:
                score += 20
            if self.morning_star(data).iloc[-1]:
                score += 25
            if self.breakout(data)[0].iloc[-1]:
                score += 20
        except Exception as e:
            logger.warning(f"Error calculating pattern score: {e}")

        return score

    def process(self, data: pd.DataFrame):
        if data.empty:
            return data, 0

        required = ["Open", "High", "Low", "Close"]
        if not all(col in data.columns for col in required):
            logger.error("Missing required OHLC columns")
            return data, 0

        data = data.copy()

        try:
            data["BULLISH_ENGULFING"] = self.bullish_engulfing(data)
            data["BEARISH_ENGULFING"] = self.bearish_engulfing(data)
            data["HAMMER"] = self.hammer(data)
            data["SHOOTING_STAR"] = self.shooting_star(data)
            data["DOJI"] = self.doji(data)
            data["MORNING_STAR"] = self.morning_star(data)
            data["EVENING_STAR"] = self.evening_star(data)
            
            # Breakout columns
            up, down = self.breakout(data)
            data["BREAKOUT_UP"] = up
            data["BREAKOUT_DOWN"] = down
            
        except Exception as e:
            logger.error(f"Error adding pattern columns: {e}")

        score = self.pattern_score(data)
        return data, score