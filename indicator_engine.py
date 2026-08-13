import pandas as pd
import numpy as np

from config import *


class IndicatorEngine:
    """Computes technical indicators and scores."""

    def __init__(self):
        pass

    # ---------- EMA ----------
    def ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        return data["Close"].ewm(span=period, adjust=False).mean()

    # ---------- RSI ----------
    def rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        # Prevent division by zero
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # neutral when no movement
        return rsi

    # ---------- ATR ----------
    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ---------- MACD ----------
    def macd(self, data: pd.DataFrame):
        ema12 = data["Close"].ewm(span=MACD_FAST, adjust=False).mean()
        ema26 = data["Close"].ewm(span=MACD_SLOW, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal

        return macd_line, signal, histogram

    # ---------- VWAP ----------
    def vwap(self, data: pd.DataFrame) -> pd.Series:
        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        vol_cum = data["Volume"].cumsum().replace(0, 1)
        return (tp * data["Volume"]).cumsum() / vol_cum

    # ---------- Volume ----------
    def volume_ma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        return data["Volume"].rolling(period).mean()

    def volume_spike(self, data: pd.DataFrame, vol_ma: pd.Series = None) -> pd.Series:
        if vol_ma is None:
            vol_ma = self.volume_ma(data)
        return data["Volume"] > (vol_ma * 1.5)

    # ---------- Trend Score (per-row, vectorized) ----------
    def trend_score_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute TrendScore for EVERY row using only that row's own
        values (no look-ahead). Replaces the old scalar-broadcast
        version which incorrectly assigned the LAST row's score to
        the entire history.
        """
        if data.empty:
            return pd.Series(dtype=float)

        ema20 = data.get("EMA20", pd.Series(0, index=data.index))
        ema50 = data.get("EMA50", pd.Series(0, index=data.index))
        close = data.get("Close", pd.Series(0, index=data.index))
        vwap = data.get("VWAP", pd.Series(0, index=data.index))
        macd = data.get("MACD", pd.Series(0, index=data.index))
        macd_signal = data.get("MACD_SIGNAL", pd.Series(0, index=data.index))
        vol_spike = data.get("VOL_SPIKE", pd.Series(False, index=data.index))
        rsi = data.get("RSI", pd.Series(50, index=data.index))

        score = pd.Series(0, index=data.index, dtype=float)
        score += (ema20 > ema50).astype(int) * 30
        score += (close > vwap).astype(int) * 20
        score += (macd > macd_signal).astype(int) * 20
        score += vol_spike.astype(bool).astype(int) * 15
        score += (rsi > 55).astype(int) * 15
        return score

    def trend_score(self, data: pd.DataFrame) -> int:
        """Trend score for the most recent (current) candle only."""
        series = self.trend_score_series(data)
        if series.empty:
            return 0
        return int(series.iloc[-1])

    # ---------- Momentum Score (per-row, vectorized) ----------
    def momentum_score_series(self, data: pd.DataFrame) -> pd.Series:
        """Compute MomentumScore for EVERY row using only that row's own values."""
        if data.empty:
            return pd.Series(dtype=float)

        rsi = data.get("RSI", pd.Series(50, index=data.index))
        macd_hist = data.get("MACD_HIST", pd.Series(0, index=data.index))
        atr = data.get("ATR", pd.Series(0, index=data.index))
        # Rolling mean uses only past + current values at each point - no look-ahead.
        atr_ma = atr.rolling(20, min_periods=1).mean()

        score = pd.Series(0, index=data.index, dtype=float)
        score += ((rsi >= 55) & (rsi <= 70)).astype(int) * 40
        score += (macd_hist > 0).astype(int) * 30
        score += (atr > atr_ma).astype(int) * 30
        return score

    def momentum_score(self, data: pd.DataFrame) -> int:
        """Momentum score for the most recent (current) candle only."""
        series = self.momentum_score_series(data)
        if series.empty:
            return 0
        return int(series.iloc[-1])

    # ---------- Final Score ----------
    def final_score(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {"trend_score": 0, "momentum_score": 0, "total_score": 0}

        trend = self.trend_score(data)
        momentum = self.momentum_score(data)
        return {
            "trend_score": trend,
            "momentum_score": momentum,
            "total_score": trend + momentum
        }

    # ---------- Process ----------
    def process(self, data: pd.DataFrame):
        if data.empty:
            return data, {}

        data = data.copy()

        # Basic indicators
        data["EMA20"] = self.ema(data, EMA_FAST)
        data["EMA50"] = self.ema(data, EMA_SLOW)
        data["RSI"] = self.rsi(data, RSI_PERIOD)
        data["ATR"] = self.atr(data, ATR_PERIOD)

        # Advanced indicators
        macd, signal, hist = self.macd(data)
        data["MACD"] = macd
        data["MACD_SIGNAL"] = signal
        data["MACD_HIST"] = hist

        data["VWAP"] = self.vwap(data)

        data["VOL_MA"] = self.volume_ma(data)
        data["VOL_SPIKE"] = self.volume_spike(data, vol_ma=data["VOL_MA"])

        # Trend / momentum score (per-row, no look-ahead)
        data["TrendScore"] = self.trend_score_series(data)
        data["MomentumScore"] = self.momentum_score_series(data)

        # Fill remaining NaNs with modern methods
        data = data.bfill().ffill()

        score = self.final_score(data)
        return data, score