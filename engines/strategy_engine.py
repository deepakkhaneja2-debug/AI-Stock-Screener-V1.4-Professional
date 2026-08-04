import logging
import pandas as pd
from typing import Dict, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StrategyEngine:
    """
    Combines multiple strategies with improved weighting.
    Enhanced to reduce false signals using confirmation logic.
    """

    def __init__(self):
        self.max_score_per_strategy = 25
        self.buy_threshold = 80
        self.sell_threshold = 80
        self.confirmation_candles = 5
        
        # Minimum requirements for BUY
        self.min_ema_trend = 20
        self.min_macd_trend = 20
        self.min_total_positive = 40

    def _safe_get_last(self, data: pd.DataFrame, column: str, default: float = 0.0) -> float:
        if data.empty or column not in data.columns:
            return default
        try:
            val = data[column].iloc[-1]
            return float(val) if pd.notna(val) else default
        except Exception:
            return default

    def _safe_get_bool(self, data: pd.DataFrame, column: str) -> bool:
        if data.empty or column not in data.columns:
            return False
        try:
            val = data[column].iloc[-1]
            return bool(val) if pd.notna(val) else False
        except Exception:
            return False

    def _confirm_signal(self, data: pd.DataFrame, column: str, direction: str) -> bool:
        """Check if signal persists for confirmation candles."""
        if data.empty or column not in data.columns:
            return False
        try:
            last_n = data[column].iloc[-self.confirmation_candles:]
            if direction == "bullish":
                return all(last_n > 0)
            elif direction == "bearish":
                return all(last_n < 0)
            return False
        except Exception:
            return False

    def _ema_trend(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        ema20 = self._safe_get_last(data, "EMA20")
        ema50 = self._safe_get_last(data, "EMA50")
        close = self._safe_get_last(data, "Close")
        
        if ema20 == 0 or ema50 == 0:
            return 0
            
        if ema20 > ema50 and close > ema20:
            if self._confirm_signal(data, "EMA20", "bullish"):
                return 25
        elif ema20 < ema50 and close < ema20:
            if self._confirm_signal(data, "EMA20", "bearish"):
                return -25
        return 0

    def _macd(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        macd = self._safe_get_last(data, "MACD")
        signal = self._safe_get_last(data, "MACD_SIGNAL")
        hist = self._safe_get_last(data, "MACD_HIST", 0)
        
        if macd > signal and hist > 0:
            if self._confirm_signal(data, "MACD", "bullish"):
                return 25
        elif macd < signal and hist < 0:
            if self._confirm_signal(data, "MACD", "bearish"):
                return -25
        return 0

    def _rsi(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        rsi = self._safe_get_last(data, "RSI")
        if rsi == 0:
            return 0
            
        if 58 <= rsi <= 72:
            return 20
        elif 28 <= rsi <= 42:
            return -20
        return 0

    def _volume_spike(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        spike = self._safe_get_bool(data, "VOL_SPIKE")
        if not spike:
            return 0
        close = self._safe_get_last(data, "Close")
        vwap = self._safe_get_last(data, "VWAP")
        
        if close > vwap * 1.01:
            return 15
        elif close < vwap * 0.99:
            return -15
        return 0

    def _pattern(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        bullish = ["BULLISH_ENGULFING", "HAMMER", "MORNING_STAR"]
        bearish = ["BEARISH_ENGULFING", "SHOOTING_STAR", "EVENING_STAR"]
        
        for pat in bullish:
            if self._safe_get_bool(data, pat):
                return 15
        for pat in bearish:
            if self._safe_get_bool(data, pat):
                return -15
        return 0

    def _breakout(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        up = self._safe_get_bool(data, "BREAKOUT_UP")
        down = self._safe_get_bool(data, "BREAKOUT_DOWN")
        vol_spike = self._safe_get_bool(data, "VOL_SPIKE")
        
        if up and vol_spike:
            if self._confirm_signal(data, "BREAKOUT_UP", "bullish"):
                return 15
        elif down and vol_spike:
            if self._confirm_signal(data, "BREAKOUT_DOWN", "bearish"):
                return -15
        return 0

    def _trend_score(self, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        trend = self._safe_get_last(data, "TrendScore")
        if trend >= 70:
            return 20
        elif trend <= 30:
            return -20
        return 0

    def _market_filter(self, data: pd.DataFrame) -> bool:
        """Check market condition before taking BUY."""
        if data.empty:
            return False
        try:
            close = self._safe_get_last(data, "Close")
            sma50 = self._safe_get_last(data, "SMA50")
            sma200 = self._safe_get_last(data, "SMA200")
            
            if sma50 == 0 or sma200 == 0:
                return True
                
            if close > sma50 and close > sma200:
                return True
            return False
        except:
            return True

    def evaluate(self, data: pd.DataFrame) -> Dict[str, Union[str, int, List[str]]]:
        if data.empty:
            return {"signal": "WATCH", "strategy_score": 0, "triggered_strategies": []}

        required = ["Close", "EMA20", "EMA50", "MACD", "MACD_SIGNAL",
                    "RSI", "VOL_SPIKE", "VWAP", "TrendScore", "SMA50", "SMA200"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return {"signal": "WATCH", "strategy_score": 0, "triggered_strategies": []}

        try:
            strategies = {
                "EMA_Trend": self._ema_trend(data),
                "MACD_Momentum": self._macd(data),
                "RSI": self._rsi(data),
                "Volume_Spike": self._volume_spike(data),
                "Candlestick_Pattern": self._pattern(data),
                "Breakout": self._breakout(data),
                "Trend_Score": self._trend_score(data),
            }

            total = sum(strategies.values())
            market_ok = self._market_filter(data)
            
            ema_score = strategies.get("EMA_Trend", 0)
            macd_score = strategies.get("MACD_Momentum", 0)
            positive_score = sum(v for v in strategies.values() if v > 0)
            
            max_total = len(strategies) * 25
            norm_score = ((total + max_total) / (2 * max_total)) * 100
            norm_score = max(0, min(100, round(norm_score)))

            # STRICT BUY CONDITION
            if (norm_score >= self.buy_threshold and 
                market_ok and 
                ema_score >= self.min_ema_trend and 
                macd_score >= self.min_macd_trend and
                positive_score >= self.min_total_positive):
                signal = "BUY"
            elif (norm_score >= self.sell_threshold and 
                  ema_score <= -self.min_ema_trend and 
                  macd_score <= -self.min_macd_trend):
                signal = "SELL"
            else:
                signal = "WATCH"

            if signal == "BUY":
                triggered = [n for n, s in strategies.items() if s > 0]
            elif signal == "SELL":
                triggered = [n for n, s in strategies.items() if s < 0]
            else:
                triggered = [n for n, s in strategies.items() if s != 0]

            logger.info(f"Signal: {signal}, Score: {norm_score}, Triggered: {triggered}")
            return {
                "signal": signal,
                "strategy_score": norm_score,
                "triggered_strategies": triggered
            }

        except Exception as e:
            logger.error(f"Error in evaluate: {e}", exc_info=True)
            return {"signal": "WATCH", "strategy_score": 0, "triggered_strategies": []}