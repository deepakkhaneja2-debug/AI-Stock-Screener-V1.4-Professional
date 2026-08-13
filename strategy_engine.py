import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StrategyEngine:
    """
    Institutional Grade Strategy Engine with AI Score Integration.
    Combines multiple strategies with dynamic weighting and confidence scoring.
    Optimized for production performance and accuracy.
    """

    def __init__(self):
        self.max_score_per_strategy = 20
        self.buy_threshold = 55
        self.sell_threshold = 45
        self.confirmation_candles = 3
        
        # AI Score weights - Institutional optimized
        self.strategy_weight = 0.40
        self.pattern_weight = 0.15
        self.risk_weight = 0.25
        self.backtest_weight = 0.20
        
        # Confidence thresholds - Institutional grade
        self.high_confidence_threshold = 75
        self.medium_confidence_threshold = 60
        self.low_confidence_threshold = 45
        
        # Caching for performance
        self._cache = {}
        self._cache_size = 100
        self._cache_timestamp = None

    def _get_cache_key(self, column: str, n: int, data: pd.DataFrame) -> str:
        """Generate cache key based on dataframe timestamp/index to prevent stale cache."""
        try:
            # Use the last index value as timestamp
            if len(data) > 0:
                last_index = data.index[-1]
                # Convert to string to handle different index types
                timestamp = str(last_index)
            else:
                timestamp = str(len(data))
            return f"{column}_{n}_{timestamp}"
        except Exception:
            # Fallback to length-based key if index access fails
            return f"{column}_{n}_{len(data)}"

    def _get_cached(self, key: str, data: pd.DataFrame) -> Optional[pd.Series]:
        """Get cached series if available."""
        if key in self._cache:
            return self._cache[key]
        return None

    def _set_cache(self, key: str, value: pd.Series):
        """Set cached series with size limit."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def _safe_get_last(self, data: pd.DataFrame, column: str, default: float = 0.0, offset: int = 0) -> float:
        """Safely get last value from dataframe column with offset."""
        if data.empty or column not in data.columns:
            return default
        try:
            if len(data) <= offset:
                return default
            idx = -1 - offset
            val = data[column].iloc[idx]
            return float(val) if pd.notna(val) else default
        except (IndexError, ValueError, TypeError):
            return default

    def _safe_get_bool(self, data: pd.DataFrame, column: str, offset: int = 0) -> bool:
        """Safely get boolean value from dataframe column."""
        if data.empty or column not in data.columns:
            return False
        try:
            if len(data) <= offset:
                return False
            idx = -1 - offset
            val = data[column].iloc[idx]
            return bool(val) if pd.notna(val) else False
        except (IndexError, ValueError, TypeError):
            return False

    def _get_series_slice(self, data: pd.DataFrame, column: str, n: int) -> Optional[pd.Series]:
        """Get last n values from a column with caching."""
        if data.empty or column not in data.columns or len(data) < n:
            return None
        try:
            cache_key = self._get_cache_key(column, n, data)
            cached = self._get_cached(cache_key, data)
            if cached is not None:
                return cached
            series = data[column].iloc[-n:].copy()
            self._set_cache(cache_key, series)
            return series
        except Exception:
            return None

    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope using vectorized operations."""
        if series is None or len(series) < 3:
            return 0.0
        try:
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return float(slope)
        except Exception:
            return 0.0

    def _calculate_rsi_slope(self, data: pd.DataFrame) -> float:
        """Optimized RSI slope calculation."""
        rsi_series = self._get_series_slice(data, "RSI", 5)
        return self._calculate_slope(rsi_series)

    def _check_ema_expansion(self, data: pd.DataFrame, ema1: str, ema2: str, n: int = 3) -> bool:
        """Check if EMA distance is expanding."""
        try:
            if len(data) < n:
                return False
            dist_series = abs(data[ema1].iloc[-n:] - data[ema2].iloc[-n:])
            if len(dist_series) > 1:
                return float(dist_series.iloc[-1]) > float(dist_series.iloc[0])
            return False
        except Exception:
            return False

    def _check_ema_compression(self, data: pd.DataFrame, ema1: str, ema2: str, n: int = 5) -> bool:
        """Check if EMA distance is compressing."""
        try:
            if len(data) < n:
                return False
            dist_series = abs(data[ema1].iloc[-n:] - data[ema2].iloc[-n:])
            if len(dist_series) > 1:
                return float(dist_series.iloc[-1]) < float(dist_series.iloc[0])
            return False
        except Exception:
            return False

    def _confirm_signal(self, data: pd.DataFrame, column: str, direction: str) -> bool:
        """Multi-candle confirmation with proper logic."""
        if data.empty or column not in data.columns or len(data) < self.confirmation_candles:
            return False
        try:
            last_n = data[column].iloc[-self.confirmation_candles:]
            
            if direction == "bullish":
                if all(last_n > 0):
                    return True
                return False
            elif direction == "bearish":
                if all(last_n < 0):
                    return True
                return False
            return False
        except Exception:
            return False

    # ---------- Individual Strategies ----------

    def _ema_trend(self, data: pd.DataFrame) -> int:
        """Institutional EMA Strategy with slope, distance, and expansion analysis."""
        if data.empty or len(data) < 5:
            return 0
            
        ema20 = self._safe_get_last(data, "EMA20")
        ema50 = self._safe_get_last(data, "EMA50")
        close = self._safe_get_last(data, "Close")
        
        if ema20 == 0 or ema50 == 0 or close == 0:
            return 0

        # Calculate slopes using cached series
        ema20_series = self._get_series_slice(data, "EMA20", 5)
        ema50_series = self._get_series_slice(data, "EMA50", 5)
        close_series = self._get_series_slice(data, "Close", 5)
        
        ema20_slope = self._calculate_slope(ema20_series)
        ema50_slope = self._calculate_slope(ema50_series)
        close_slope = self._calculate_slope(close_series)
        
        # Calculate distance percentage - vectorized
        ema_distance = abs(ema20 - ema50)
        avg_price = (ema20 + ema50) / 2
        distance_pct = (ema_distance / avg_price) * 100 if avg_price > 0 else 0
        
        # Check expansion/compression
        expanding = self._check_ema_expansion(data, "EMA20", "EMA50")
        compressing = self._check_ema_compression(data, "EMA20", "EMA50")
        
        score = 0
        trend_strength = 0
        
        # Bullish conditions
        if ema20 > ema50 and close > ema20:
            # Slope confirmation with weighted scoring
            if ema20_slope > 0 and ema50_slope > 0 and close_slope > 0:
                trend_strength = 3
            elif ema20_slope > 0 and close_slope > 0:
                trend_strength = 2
            elif ema20_slope > 0:
                trend_strength = 1
            else:
                trend_strength = 0
                
            # Distance scoring with institutional thresholds
            if distance_pct > 3.0:
                trend_strength += 2
            elif distance_pct > 1.5:
                trend_strength += 1
                
            # Expansion bonus
            if expanding:
                trend_strength += 2
            elif compressing:
                trend_strength -= 1
                
            # Cap at max score
            score = min(20, 10 + trend_strength * 3)
            if score < 0:
                score = 0
                
        # Bearish conditions
        elif ema20 < ema50 and close < ema20:
            if ema20_slope < 0 and ema50_slope < 0 and close_slope < 0:
                trend_strength = -3
            elif ema20_slope < 0 and close_slope < 0:
                trend_strength = -2
            elif ema20_slope < 0:
                trend_strength = -1
            else:
                trend_strength = 0
                
            if distance_pct > 3.0:
                trend_strength -= 2
            elif distance_pct > 1.5:
                trend_strength -= 1
                
            if expanding:
                trend_strength -= 2
            elif compressing:
                trend_strength += 1
                
            score = max(-20, -10 + trend_strength * 3)
            if score > 0:
                score = 0
                
        return score

    def _macd(self, data: pd.DataFrame) -> int:
        """Institutional MACD Strategy with crossover, histogram, and zero-line analysis."""
        if data.empty or len(data) < 5:
            return 0
            
        macd = self._safe_get_last(data, "MACD")
        signal = self._safe_get_last(data, "MACD_SIGNAL")
        hist = self._safe_get_last(data, "MACD_HIST", 0)
        
        if macd == 0 or signal == 0:
            return 0

        # Multi-candle histogram analysis
        hist_series = self._get_series_slice(data, "MACD_HIST", self.confirmation_candles)
        hist_increasing = False
        hist_decreasing = False
        
        if hist_series is not None and len(hist_series) >= 3:
            hist_increasing = all(hist_series.iloc[i] < hist_series.iloc[i+1] for i in range(len(hist_series)-1))
            hist_decreasing = all(hist_series.iloc[i] > hist_series.iloc[i+1] for i in range(len(hist_series)-1))
        
        # Fresh crossover detection
        prev_macd = self._safe_get_last(data, "MACD", 0, offset=1)
        prev_signal = self._safe_get_last(data, "MACD_SIGNAL", 0, offset=1)
        
        bullish_crossover = False
        bearish_crossover = False
        
        if prev_macd != 0 and prev_signal != 0:
            bullish_crossover = (prev_macd < prev_signal) and (macd > signal)
            bearish_crossover = (prev_macd > prev_signal) and (macd < signal)
        
        # Zero-line confirmation
        above_zero = macd > 0
        below_zero = macd < 0
        
        score = 0
        momentum_strength = 0
        
        # Bullish conditions with institutional scoring
        if macd > signal:
            if hist > 0 and hist_increasing:
                momentum_strength += 2
            if bullish_crossover:
                momentum_strength += 3
            if above_zero:
                momentum_strength += 1
                
            score = min(20, 10 + momentum_strength * 3)
            if score < 0:
                score = 0
                
        # Bearish conditions
        elif macd < signal:
            if hist < 0 and hist_decreasing:
                momentum_strength -= 2
            if bearish_crossover:
                momentum_strength -= 3
            if below_zero:
                momentum_strength -= 1
                
            score = max(-20, -10 + momentum_strength * 3)
            if score > 0:
                score = 0
                
        return score

    def _rsi(self, data: pd.DataFrame) -> int:
        """Institutional RSI Strategy with slope and zone optimization."""
        if data.empty:
            return 0
            
        rsi = self._safe_get_last(data, "RSI")
        if rsi == 0:
            return 0

        # RSI slope calculation
        rsi_slope = self._calculate_rsi_slope(data)
        
        # Multi-candle confirmation
        rsi_confirmation = self._get_series_slice(data, "RSI", 3)
        rsi_increasing = False
        rsi_decreasing = False
        
        if rsi_confirmation is not None and len(rsi_confirmation) >= 3:
            rsi_increasing = all(rsi_confirmation.iloc[i] < rsi_confirmation.iloc[i+1] for i in range(len(rsi_confirmation)-1))
            rsi_decreasing = all(rsi_confirmation.iloc[i] > rsi_confirmation.iloc[i+1] for i in range(len(rsi_confirmation)-1))

        score = 0
        
        # Institutional RSI zones for swing trading
        if 55 <= rsi <= 68:  # Strong bullish zone
            score = 15
            if rsi_slope > 0 and rsi_increasing:
                score = 20
                
        elif 45 < rsi < 55:  # Neutral zone - trend continuation
            if rsi_slope > 0 and rsi_increasing:
                score = 12
            elif rsi_slope < 0 and rsi_decreasing:
                score = -12
            else:
                score = 0
                
        elif 68 < rsi <= 75:  # Overbought zone - caution
            if rsi_slope < 0 and rsi_decreasing:
                score = 0  # Wait for reversal confirmation
            else:
                score = 5  # Weak bullish, use caution
                
        elif 32 <= rsi <= 45:  # Strong bearish zone
            score = -15
            if rsi_slope < 0 and rsi_decreasing:
                score = -20
                
        elif 25 <= rsi < 32:  # Oversold zone - caution
            if rsi_slope > 0 and rsi_increasing:
                score = 0  # Wait for reversal confirmation
            else:
                score = -5  # Weak bearish, use caution
                
        return score

    def _volume_spike(self, data: pd.DataFrame) -> int:
        """Enhanced volume spike with confirmation."""
        if data.empty:
            return 0
            
        spike = self._safe_get_bool(data, "VOL_SPIKE")
        if not spike:
            return 0
            
        close = self._safe_get_last(data, "Close")
        vwap = self._safe_get_last(data, "VWAP")
        
        if close == 0 or vwap == 0:
            return 0
        
        # Volume trend confirmation
        vol_series = self._get_series_slice(data, "Volume", 5)
        vol_increasing = False
        
        if vol_series is not None and len(vol_series) >= 3:
            vol_increasing = all(vol_series.iloc[i] < vol_series.iloc[i+1] for i in range(len(vol_series)-1))
        
        # Multi-candle volume confirmation
        vol_spike_confirm = all(self._safe_get_bool(data, "VOL_SPIKE", offset=i) for i in range(self.confirmation_candles))
        
        score = 0
        if close > vwap:
            if vol_increasing or vol_spike_confirm:
                score = 20
            else:
                score = 15
        elif close < vwap:
            if vol_increasing or vol_spike_confirm:
                score = -20
            else:
                score = -15
                
        return score

    def _pattern(self, data: pd.DataFrame) -> int:
        """Institutional Pattern Strategy with weighted scoring and diminishing returns."""
        if data.empty:
            return 0
            
        # Institutional pattern weights based on reliability
        pattern_weights = {
            "MORNING_STAR": 20,
            "HAMMER": 18,
            "BULLISH_ENGULFING": 16,
            "PIERCING_LINE": 15,
            "INSIDE_BAR": 12,
            "THREE_WHITE_SOLDIERS": 18,
            "EVENING_STAR": -20,
            "SHOOTING_STAR": -18,
            "BEARISH_ENGULFING": -16,
            "DARK_CLOUD_COVER": -15,
            "BEARISH_INSIDE_BAR": -12,
            "THREE_BLACK_CROWS": -18,
        }
        
        bullish_patterns = ["MORNING_STAR", "HAMMER", "BULLISH_ENGULFING", "PIERCING_LINE", "INSIDE_BAR", "THREE_WHITE_SOLDIERS"]
        bearish_patterns = ["EVENING_STAR", "SHOOTING_STAR", "BEARISH_ENGULFING", "DARK_CLOUD_COVER", "BEARISH_INSIDE_BAR", "THREE_BLACK_CROWS"]
        
        # Score accumulation with diminishing returns for multiple patterns
        bullish_score = 0
        bearish_score = 0
        pattern_count = 0
        max_pattern_score = 20
        
        # Check all bullish patterns with diminishing returns
        for pattern in bullish_patterns:
            if pattern in data.columns and self._safe_get_bool(data, pattern):
                weight = pattern_weights.get(pattern, 15)
                # Apply diminishing returns: each additional pattern adds less
                if pattern_count == 0:
                    bullish_score += weight
                elif pattern_count == 1:
                    bullish_score += weight * 0.7  # 30% reduction for second pattern
                elif pattern_count == 2:
                    bullish_score += weight * 0.5  # 50% reduction for third pattern
                else:
                    bullish_score += weight * 0.3  # 70% reduction for more patterns
                pattern_count += 1
                
        pattern_count = 0
        
        # Check all bearish patterns with diminishing returns
        for pattern in bearish_patterns:
            if pattern in data.columns and self._safe_get_bool(data, pattern):
                weight = abs(pattern_weights.get(pattern, -15))
                if pattern_count == 0:
                    bearish_score += weight
                elif pattern_count == 1:
                    bearish_score += weight * 0.7
                elif pattern_count == 2:
                    bearish_score += weight * 0.5
                else:
                    bearish_score += weight * 0.3
                pattern_count += 1
                
        # Return net score with cap to prevent inflation
        if bullish_score > 0 and bearish_score == 0:
            return min(max_pattern_score, int(bullish_score))
        elif bearish_score > 0 and bullish_score == 0:
            return max(-max_pattern_score, -int(bearish_score))
        elif bullish_score > 0 and bearish_score > 0:
            net_score = bullish_score - bearish_score
            if net_score > 0:
                return min(max_pattern_score, int(net_score))
            else:
                return max(-max_pattern_score, int(net_score))
            
        return 0

    def _breakout(self, data: pd.DataFrame) -> int:
        """Institutional Breakout Strategy with volume, ATR, and false breakout filters."""
        if data.empty:
            return 0
            
        up = self._safe_get_bool(data, "BREAKOUT_UP")
        down = self._safe_get_bool(data, "BREAKOUT_DOWN")
        vol_spike = self._safe_get_bool(data, "VOL_SPIKE")
        
        # Fresh breakout detection
        prev_up = self._safe_get_bool(data, "BREAKOUT_UP", offset=1)
        prev_down = self._safe_get_bool(data, "BREAKOUT_DOWN", offset=1)
        fresh_breakout = (up and not prev_up) or (down and not prev_down)
        
        # ATR expansion confirmation
        atr = self._safe_get_last(data, "ATR", 0)
        atr_prev = self._safe_get_last(data, "ATR", 0, offset=1)
        atr_expanding = atr > atr_prev if atr_prev > 0 else False
        
        # Volume confirmation
        volume = self._safe_get_last(data, "Volume", 0)
        if volume == 0:
            return 0
            
        volume_series = self._get_series_slice(data, "Volume", 20)
        if volume_series is None or len(volume_series) < 20:
            return 0
            
        volume_avg = float(volume_series.mean())
        volume_surge = volume > (volume_avg * 1.5) if volume_avg > 0 else False
        
        score = 0
        
        if up:
            if vol_spike and fresh_breakout and atr_expanding:
                score = 20
            elif vol_spike and fresh_breakout:
                score = 16
            elif vol_spike and atr_expanding:
                score = 12
            elif volume_surge and fresh_breakout:
                score = 10
            else:
                score = 0  # False breakout filter
                
        elif down:
            if vol_spike and fresh_breakout and atr_expanding:
                score = -20
            elif vol_spike and fresh_breakout:
                score = -16
            elif vol_spike and atr_expanding:
                score = -12
            elif volume_surge and fresh_breakout:
                score = -10
            else:
                score = 0
                
        return score

    def _trend_score(self, data: pd.DataFrame) -> int:
        """Institutional TrendScore with fixed logical order."""
        if data.empty:
            return 0
        trend = self._safe_get_last(data, "TrendScore")
        
        # Fixed order: higher values first
        if trend >= 70:
            return 20
        elif trend >= 60:
            return 15
        elif trend <= 30:
            return -20
        elif trend <= 40:
            return -15
        return 0

    def _adx_trend(self, data: pd.DataFrame) -> int:
        """ADX Trend Strength Strategy."""
        if data.empty:
            return 0
            
        adx = self._safe_get_last(data, "ADX", 0)
        if adx == 0:
            return 0
            
        di_plus = self._safe_get_last(data, "DI_PLUS", 0)
        di_minus = self._safe_get_last(data, "DI_MINUS", 0)
        
        if di_plus == 0 or di_minus == 0:
            return 0
        
        # ADX slope
        adx_series = self._get_series_slice(data, "ADX", 5)
        adx_slope = self._calculate_slope(adx_series)
        
        score = 0
        
        # Strong trend with ADX
        if adx > 25:
            if di_plus > di_minus:
                score = 15
                if adx_slope > 0:
                    score = 20
            elif di_minus > di_plus:
                score = -15
                if adx_slope < 0:
                    score = -20
        elif adx > 20:
            if di_plus > di_minus:
                score = 10  # Moderate bullish
            elif di_minus > di_plus:
                score = -10  # Moderate bearish
                
        return score

    def _supertrend(self, data: pd.DataFrame) -> int:
        """Supertrend Confirmation Strategy."""
        if data.empty:
            return 0
            
        supertrend = self._safe_get_last(data, "SUPERTREND", 0)
        if supertrend == 0:
            return 0
            
        # Previous value for fresh signal detection
        prev_supertrend = self._safe_get_last(data, "SUPERTREND", 0, offset=1)
        
        fresh_signal = False
        if prev_supertrend != 0:
            fresh_signal = (supertrend > 0 and prev_supertrend <= 0) or (supertrend < 0 and prev_supertrend >= 0)
        
        score = 0
        if supertrend > 0:  # Bullish
            score = 20 if fresh_signal else 15
        elif supertrend < 0:  # Bearish
            score = -20 if fresh_signal else -15
            
        return score

    def _atr_trend(self, data: pd.DataFrame) -> int:
        """ATR Trend Strategy."""
        if data.empty:
            return 0
            
        atr = self._safe_get_last(data, "ATR", 0)
        if atr == 0:
            return 0
            
        atr_series = self._get_series_slice(data, "ATR", 5)
        atr_slope = self._calculate_slope(atr_series)
        
        # Determine trend direction from price
        close = self._safe_get_last(data, "Close", 0)
        close_prev = self._safe_get_last(data, "Close", 0, offset=self.confirmation_candles)
        
        if close == 0 or close_prev == 0:
            return 0
        
        price_change = close - close_prev
        price_pct_change = (price_change / close_prev) * 100 if close_prev > 0 else 0
        
        score = 0
        
        if price_pct_change > 1.0 and atr_slope > 0:
            score = 15  # Bullish with increasing volatility
        elif price_pct_change < -1.0 and atr_slope < 0:
            score = -15  # Bearish with increasing volatility
        elif price_pct_change > 0.5 and atr_slope > 0:
            score = 10  # Moderate bullish
        elif price_pct_change < -0.5 and atr_slope < 0:
            score = -10  # Moderate bearish
        elif price_pct_change > 0 and atr_slope < 0:
            score = 5  # Bullish but volatility decreasing
        elif price_pct_change < 0 and atr_slope > 0:
            score = -5  # Bearish but volatility decreasing
            
        return score

    def _support_resistance(self, data: pd.DataFrame) -> int:
        """Support Resistance Score Strategy."""
        if data.empty:
            return 0
            
        support = self._safe_get_last(data, "SUPPORT", 0)
        resistance = self._safe_get_last(data, "RESISTANCE", 0)
        close = self._safe_get_last(data, "Close", 0)
        
        if support == 0 or resistance == 0 or close == 0:
            return 0
            
        # Calculate distance to levels with percentages
        support_distance = abs(close - support) / support * 100 if support > 0 else 0
        resistance_distance = abs(close - resistance) / resistance * 100 if resistance > 0 else 0
        
        score = 0
        
        # Near support (bullish)
        if support_distance < 2.0 and close > support:
            score = 15
            # Check if bounced from support
            if self._safe_get_last(data, "Close", 0, offset=1) < support:
                score = 20  # Fresh bounce
                
        # Near resistance (bearish)
        elif resistance_distance < 2.0 and close < resistance:
            score = -15
            if self._safe_get_last(data, "Close", 0, offset=1) > resistance:
                score = -20  # Fresh rejection
                
        # Strong support zone
        elif support_distance < 5.0 and close > support:
            score = 10
            
        # Strong resistance zone
        elif resistance_distance < 5.0 and close < resistance:
            score = -10
            
        return score

    def _vwap_confirmation(self, data: pd.DataFrame) -> int:
        """VWAP Confirmation Strategy."""
        if data.empty:
            return 0
            
        close = self._safe_get_last(data, "Close", 0)
        vwap = self._safe_get_last(data, "VWAP", 0)
        
        if vwap == 0 or close == 0:
            return 0
            
        # VWAP slope
        vwap_series = self._get_series_slice(data, "VWAP", 5)
        vwap_slope = self._calculate_slope(vwap_series)
        
        # Price relative to VWAP
        vwap_distance = (close - vwap) / vwap * 100 if vwap > 0 else 0
        
        score = 0
        
        if close > vwap and vwap_slope > 0:
            score = 15
            if vwap_distance > 2.0:
                score = 20  # Strong bullish
        elif close > vwap and vwap_slope < 0:
            score = 10
        elif close < vwap and vwap_slope < 0:
            score = -15
            if vwap_distance < -2.0:
                score = -20  # Strong bearish
        elif close < vwap and vwap_slope > 0:
            score = -10
            
        return score

    def _hh_hl(self, data: pd.DataFrame) -> int:
        """Higher High Higher Low Strategy."""
        if data.empty or len(data) < 5:
            return 0
            
        # Get last 5 candles
        high_series = self._get_series_slice(data, "High", 5)
        low_series = self._get_series_slice(data, "Low", 5)
        
        if high_series is None or low_series is None or len(high_series) < 5:
            return 0
            
        # Check HH/HL pattern using vectorized operations
        higher_highs = all(high_series.iloc[i] < high_series.iloc[i+1] for i in range(4))
        higher_lows = all(low_series.iloc[i] < low_series.iloc[i+1] for i in range(4))
        
        # Check LH/LL pattern
        lower_highs = all(high_series.iloc[i] > high_series.iloc[i+1] for i in range(4))
        lower_lows = all(low_series.iloc[i] > low_series.iloc[i+1] for i in range(4))
        
        if higher_highs and higher_lows:
            return 20  # Strong uptrend
        elif higher_highs:
            return 15  # Weak uptrend
        elif lower_highs and lower_lows:
            return -20  # Strong downtrend
        elif lower_lows:
            return -15  # Weak downtrend
            
        return 0

    def _calculate_risk_score(self, data: pd.DataFrame) -> float:
        """Institutional Risk Score with comprehensive risk assessment."""
        try:
            # Volatility risk (ATR based)
            atr = self._safe_get_last(data, "ATR", 0)
            close = self._safe_get_last(data, "Close", 1)
            
            if close == 0:
                return 50
                
            volatility = (atr / close * 100) if close > 0 else 0
            
            # Volume consistency
            vol_series = self._get_series_slice(data, "Volume", 20)
            vol_consistency = 0
            
            if vol_series is not None and len(vol_series) > 0:
                vol_mean = float(vol_series.mean())
                vol_std = float(vol_series.std())
                if vol_mean > 0:
                    vol_consistency = 1 - (vol_std / vol_mean)
                    vol_consistency = max(0, min(1, vol_consistency))
            
            risk = 0
            
            # Volatility risk (0-40)
            if volatility > 5.0:
                risk += 35
            elif volatility > 3.0:
                risk += 25
            elif volatility > 2.0:
                risk += 15
            elif volatility > 1.0:
                risk += 5
                
            # Volume risk (0-30)
            if vol_consistency < 0.3:
                risk += 30
            elif vol_consistency < 0.5:
                risk += 20
            elif vol_consistency < 0.7:
                risk += 10
                
            # Market position risk (0-20)
            close = self._safe_get_last(data, "Close", 0)
            vwap = self._safe_get_last(data, "VWAP", 0)
            
            if close and vwap:
                vwap_deviation = (close - vwap) / vwap * 100 if vwap > 0 else 0
                if vwap_deviation < -3.0:
                    risk += 20  # Trading significantly below VWAP
                elif vwap_deviation < -2.0:
                    risk += 15
                elif vwap_deviation < -1.0:
                    risk += 10
                elif vwap_deviation > 3.0:
                    risk += 10  # Extended above VWAP
                    
            # ATR trend risk (0-10)
            atr_slope = self._calculate_slope(self._get_series_slice(data, "ATR", 5))
            if atr_slope > 0.5:
                risk += 10  # Increasing volatility
            elif atr_slope > 0.2:
                risk += 5
                
            return min(100, risk)
            
        except Exception:
            return 50

    def _get_backtest_score(self, data: pd.DataFrame) -> float:
        """Extract backtest score if available."""
        if "BACKTEST_SCORE" in data.columns:
            score = self._safe_get_last(data, "BACKTEST_SCORE", 50)
            return max(0, min(100, float(score)))
        return 50

    def _calculate_confidence(self, strategy_score: float, pattern_score: float, 
                            risk_score: float, backtest_score: float, 
                            market_regime: int = 0) -> float:
        """Institutional Confidence Score with market regime adjustment."""
        # Base confidence calculation
        base_confidence = (
            (strategy_score * 0.30) +
            (pattern_score * 0.15) +
            ((100 - risk_score) * 0.30) +
            (backtest_score * 0.25)
        )
        
        # Market regime adjustment
        if market_regime == 1:
            base_confidence *= 1.15  # Boost during strong trends
        elif market_regime == -1:
            base_confidence *= 0.85  # Reduce during choppy markets
            
        return max(0, min(100, base_confidence))

    def _detect_market_regime(self, data: pd.DataFrame) -> int:
        """
        Institutional Market Regime detection combining multiple indicators.
        Returns: 1 = Trending, 0 = Neutral, -1 = Choppy
        """
        try:
            regime_score = 0
            indicators_checked = 0
            
            # 1. ADX for trend strength
            adx = self._safe_get_last(data, "ADX", 0)
            if adx > 0:
                indicators_checked += 1
                if adx > 30:
                    regime_score += 2
                elif adx > 20:
                    regime_score += 0
                else:
                    regime_score -= 2
            
            # 2. ATR Expansion (increasing volatility can indicate trend)
            atr = self._safe_get_last(data, "ATR", 0)
            atr_prev = self._safe_get_last(data, "ATR", 0, offset=5)
            if atr > 0 and atr_prev > 0:
                indicators_checked += 1
                atr_change = ((atr - atr_prev) / atr_prev) * 100
                if atr_change > 10:
                    regime_score += 1
                elif atr_change < -10:
                    regime_score -= 1
            
            # 3. EMA20 Slope
            ema20_slope = self._calculate_slope(self._get_series_slice(data, "EMA20", 5))
            if ema20_slope != 0:
                indicators_checked += 1
                if ema20_slope > 0.5:
                    regime_score += 1
                elif ema20_slope < -0.5:
                    regime_score -= 1
            
            # 4. EMA50 Slope
            ema50_slope = self._calculate_slope(self._get_series_slice(data, "EMA50", 5))
            if ema50_slope != 0:
                indicators_checked += 1
                if ema50_slope > 0.3:
                    regime_score += 1
                elif ema50_slope < -0.3:
                    regime_score -= 1
            
            # 5. HH-HL Trend
            hh_hl_score = self._hh_hl(data)
            if hh_hl_score != 0:
                indicators_checked += 1
                if hh_hl_score > 0:
                    regime_score += 1
                else:
                    regime_score -= 1
            
            # Normalize based on indicators checked
            if indicators_checked > 0:
                normalized_score = regime_score / indicators_checked
                
                if normalized_score > 0.4:
                    return 1  # Trending
                elif normalized_score < -0.4:
                    return -1  # Choppy
                else:
                    return 0  # Neutral
            else:
                return 0
                
        except Exception:
            return 0

    # ---------- Evaluate ----------
    def evaluate(self, data: pd.DataFrame) -> Dict[str, Union[str, int, List[str], float]]:
        """Institutional evaluation with complete AI Score integration."""
        if data.empty or len(data) < 10:
            return self._get_empty_response()

        # Validate required columns
        required = ["Close", "EMA20", "EMA50", "MACD", "MACD_SIGNAL",
                    "RSI", "VOL_SPIKE", "VWAP", "TrendScore", "ATR"]
        missing = [c for c in required if c not in data.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return self._get_empty_response()

        try:
            # Clear cache for fresh evaluation
            self._cache.clear()
            
            # Calculate all strategy scores
            strategies = {
                "EMA_Trend": self._ema_trend(data),
                "MACD_Momentum": self._macd(data),
                "RSI": self._rsi(data),
                "Volume_Spike": self._volume_spike(data),
                "Candlestick_Pattern": self._pattern(data),
                "Breakout": self._breakout(data),
                "Trend_Score": self._trend_score(data),
                "ADX_Trend": self._adx_trend(data),
                "Supertrend": self._supertrend(data),
                "ATR_Trend": self._atr_trend(data),
                "Support_Resistance": self._support_resistance(data),
                "VWAP_Confirmation": self._vwap_confirmation(data),
                "HH_HL": self._hh_hl(data)
            }

            # Calculate strategy score with proper normalization
            total = sum(strategies.values())
            max_total = len(strategies) * self.max_score_per_strategy
            strategy_score = ((total + max_total) / (2 * max_total)) * 100
            strategy_score = max(0, min(100, round(strategy_score)))

            # Calculate individual component scores
            pattern_value = self._pattern(data)
            pattern_score = (abs(pattern_value) / 20) * 100
            pattern_score = max(0, min(100, pattern_score))
            
            risk_score = self._calculate_risk_score(data)
            risk_score = max(0, min(100, risk_score))
            risk_adjusted = 100 - risk_score
            
            backtest_score = self._get_backtest_score(data)
            
            # Calculate AI Score with institutional weights
            ai_score = (
                (strategy_score * self.strategy_weight) +
                (pattern_score * self.pattern_weight) +
                (risk_adjusted * self.risk_weight) +
                (backtest_score * self.backtest_weight)
            )
            ai_score = max(0, min(100, round(ai_score)))

            # Market regime detection
            market_regime = self._detect_market_regime(data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                strategy_score, pattern_score, risk_score, backtest_score, market_regime
            )
            confidence_score = round(confidence_score, 1)

            # Dynamic thresholds based on confidence and market regime
            if market_regime == 1:  # Trending market - lower threshold
                if confidence_score >= 75:
                    buy_threshold = 50
                    sell_threshold = 50
                elif confidence_score >= 60:
                    buy_threshold = 55
                    sell_threshold = 45
                else:
                    buy_threshold = 60
                    sell_threshold = 40
            else:  # Neutral or choppy - higher threshold
                if confidence_score >= 75:
                    buy_threshold = 55
                    sell_threshold = 45
                elif confidence_score >= 60:
                    buy_threshold = 60
                    sell_threshold = 40
                else:
                    buy_threshold = 65
                    sell_threshold = 35

            # Determine signal with multi-confirmation
            if ai_score >= buy_threshold:
                # Additional confirmation for BUY signals
                bullish_strategies = sum(1 for s in strategies.values() if s > 0)
                total_strategies = len(strategies)
                confirmation_ratio = bullish_strategies / total_strategies if total_strategies > 0 else 0
                
                # Require at least 30% of strategies to be bullish for BUY
                if confirmation_ratio >= 0.3 or ai_score >= 75:
                    signal = "BUY"
                else:
                    signal = "WATCH"
                    
            elif ai_score <= (100 - sell_threshold):
                bearish_strategies = sum(1 for s in strategies.values() if s < 0)
                total_strategies = len(strategies)
                confirmation_ratio = bearish_strategies / total_strategies if total_strategies > 0 else 0
                
                if confirmation_ratio >= 0.3 or ai_score <= 25:
                    signal = "SELL"
                else:
                    signal = "WATCH"
            else:
                signal = "WATCH"

            # Collect triggered strategies
            if signal == "BUY":
                triggered = [n for n, s in strategies.items() if s > 0]
            elif signal == "SELL":
                triggered = [n for n, s in strategies.items() if s < 0]
            else:
                triggered = [n for n, s in strategies.items() if s != 0]

            # Log with institutional detail
            logger.info(
                f"Signal: {signal} | AI: {ai_score} | Confidence: {confidence_score} | "
                f"Strategy: {strategy_score} | Pattern: {pattern_score:.1f} | "
                f"Risk: {risk_score:.1f} | Backtest: {backtest_score:.1f} | "
                f"Market: {'Trending' if market_regime == 1 else 'Neutral' if market_regime == 0 else 'Choppy'} | "
                f"Triggered: {len(triggered)} strategies"
            )
            
            return {
                "signal": signal,
                "strategy_score": strategy_score,
                "pattern_score": pattern_score,
                "risk_score": risk_score,
                "backtest_score": backtest_score,
                "ai_score": ai_score,
                "final_score": ai_score,
                "confidence_score": confidence_score,
                "triggered_strategies": triggered,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold
            }

        except Exception as e:
            logger.error(f"Error in evaluate: {e}", exc_info=True)
            return self._get_empty_response()

    def _get_empty_response(self) -> Dict[str, Union[str, int, List[str], float]]:
        """Return empty response with all required keys."""
        return {
            "signal": "WATCH",
            "strategy_score": 0,
            "pattern_score": 0,
            "risk_score": 50,
            "backtest_score": 50,
            "ai_score": 50,
            "final_score": 50,
            "confidence_score": 50.0,
            "triggered_strategies": [],
            "buy_threshold": 55,
            "sell_threshold": 45
        }
