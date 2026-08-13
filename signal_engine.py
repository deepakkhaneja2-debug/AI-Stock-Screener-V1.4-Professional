import pandas as pd

from config import *


class SignalEngine:

    def __init__(self):
        pass

    # ====================================
    # BUY SIGNAL
    # ====================================

    def buy_signal(self, data):

        last = data.iloc[-1]

        score = 0

        if last["EMA20"] > last["EMA50"]:
            score += 20

        if last["MACD"] > last["MACD_SIGNAL"]:
            score += 20

        if 55 < last["RSI"] < 75:
            score += 15

        if last["VOL_SPIKE"]:

            score += 15

        if last["Close"] > last["VWAP"]:
            score += 10

        if last["TrendScore"] >= 60:
            score += 20

        return score >= BUY_SCORE

    # ====================================
    # SELL SIGNAL
    # ====================================

    def sell_signal(self, data):

        last = data.iloc[-1]

        score = 0

        if last["EMA20"] < last["EMA50"]:
            score += 20

        if last["MACD"] < last["MACD_SIGNAL"]:
            score += 20

        if 25 < last["RSI"] < 45:
            score += 15

        if last["VOL_SPIKE"]:
            score += 15

        if last["Close"] < last["VWAP"]:
            score += 10

        if last["TrendScore"] <= 40:
            score += 20

        return score >= SELL_SCORE

    # ====================================
    # WATCH SIGNAL
    # ====================================

    def watch_signal(self, data):

        if self.buy_signal(data):
            return False

        if self.sell_signal(data):
            return False

        return True

        # ====================================
    # SIGNAL STRENGTH
    # ====================================

    def signal_strength(self, data):

        last = data.iloc[-1]

        strength = 0

        # EMA
        if last["EMA20"] > last["EMA50"]:
            strength += 15
        elif last["EMA20"] < last["EMA50"]:
            strength += 15

        # MACD
        if last["MACD"] > last["MACD_SIGNAL"]:
            strength += 15
        elif last["MACD"] < last["MACD_SIGNAL"]:
            strength += 15

        # RSI
        if last["RSI"] > 55:
            strength += 10
        elif last["RSI"] < 45:
            strength += 10

        # Volume
        if last["VOL_SPIKE"]:
            strength += 10

        # VWAP
        if last["Close"] > last["VWAP"]:
            strength += 10
        elif last["Close"] < last["VWAP"]:
            strength += 10

        strength += min(last["TrendScore"], 40)

        return min(strength, 100)

    # ====================================
    # FINAL SIGNAL
    # ====================================

    def generate_signal(self, data):

        if self.buy_signal(data):

            return {
                "signal": "BUY",
                "strength": self.signal_strength(data)
            }

        elif self.sell_signal(data):

            return {
                "signal": "SELL",
                "strength": self.signal_strength(data)
            }

        else:

            return {
                "signal": "WATCH",
                "strength": self.signal_strength(data)
            }
