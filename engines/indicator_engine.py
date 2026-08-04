import pandas as pd
import numpy as np

from config import *


class IndicatorEngine:
    """Computes technical indicators and scores."""

    def __init__(self):
        pass

    def ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        return data["Close"].ewm(span=period, adjust=False).mean()

    def rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi

    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def macd(self, data: pd.DataFrame):
        ema12 = data["Close"].ewm(span=MACD_FAST, adjust=False).mean()
        ema26 = data["Close"].ewm(span=MACD_SLOW, adjust=False).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal

        return macd_line, signal, histogram

    def vwap(self, data: pd.DataFrame) -> pd.Series:
        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        vol_cum = data["Volume"].cumsum().replace(0, 1)
        return (tp * data["Volume"]).cumsum() / vol_cum

    def volume_ma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        return data["Volume"].rolling(period).mean()

    def volume_spike(self, data: pd.DataFrame, vol_ma: pd.Series = None) -> pd.Series:
        if vol_ma is None:
            vol_ma = self.volume_ma(data)
        return data["Volume"] > (vol_ma * 1.5)

    def trend_score(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0

        score = 0
        last = data.iloc[-1]

        if last.get("EMA20", 0) > last.get("EMA50", 0):
            score += 30
        if last.get("Close", 0) > last.get("VWAP", 0):
            score += 20
        if last.get("MACD", 0) > last.get("MACD_SIGNAL", 0):
            score += 20
        if last.get("VOL_SPIKE", False):
            score += 15
        if last.get("RSI", 50) > 55:
            score += 15

        return score

    def momentum_score(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0

        score = 0
        last = data.iloc[-1]
        rsi = last.get("RSI", 50)

        if 58 <= rsi <= 72:
            score += 40
        if last.get("MACD_HIST", 0) > 0:
            score += 30

        atr_ma = data["ATR"].rolling(20, min_periods=1).mean()
        if last.get("ATR", 0) > atr_ma.iloc[-1]:
            score += 30

        return score

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

    def process(self, data: pd.DataFrame):
        if data.empty:
            return data, {}

        data = data.copy()

        data["EMA20"] = self.ema(data, EMA_FAST)
        data["EMA50"] = self.ema(data, EMA_SLOW)
        data["RSI"] = self.rsi(data, RSI_PERIOD)
        data["ATR"] = self.atr(data, ATR_PERIOD)

        macd, signal, hist = self.macd(data)
        data["MACD"] = macd
        data["MACD_SIGNAL"] = signal
        data["MACD_HIST"] = hist

        data["VWAP"] = self.vwap(data)

        data["VOL_MA"] = self.volume_ma(data)
        data["VOL_SPIKE"] = self.volume_spike(data, vol_ma=data["VOL_MA"])

        # 50 SMA and 200 SMA for market filter
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()

        data["TrendScore"] = self.trend_score(data)

        data = data.bfill().ffill()

        score = self.final_score(data)
        return data, score