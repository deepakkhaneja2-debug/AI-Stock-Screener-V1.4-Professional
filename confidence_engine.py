import logging
import pandas as pd

from config import *   # <-- यह import missing था, अब fix है

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConfidenceEngine:
    """Calculates confidence score from multiple weighted components."""

    def __init__(self):
        self.weight_strategy = CONFIDENCE_WEIGHT_STRATEGY
        self.weight_trend = CONFIDENCE_WEIGHT_TREND
        self.weight_pattern = CONFIDENCE_WEIGHT_PATTERN
        self.weight_volume = CONFIDENCE_WEIGHT_VOLUME
        self.weight_atr = CONFIDENCE_WEIGHT_ATR

    def _compute_atr_quality(self, atr_percent: float) -> float:
        """
        Return a quality score for ATR, based on ATR AS A PERCENTAGE OF
        PRICE (not raw currency ATR). The previous version scored raw
        ATR * 10, which is scale-dependent: identical volatility read as
        "high quality" on an expensive stock and "low quality" on a
        cheap one purely because of price level, not actual volatility.
        A moderate ATR% (comparable to config.MIN_ATR_PERCENT /
        MAX_ATR_PERCENT) scores highest; too low or too high both
        reduce quality, mirroring the backtest's ATR% filter.
        """
        if atr_percent <= 0:
            return 0.0
        min_atr = globals().get("MIN_ATR_PERCENT", 0.5)
        max_atr = globals().get("MAX_ATR_PERCENT", 6.0)
        if atr_percent < min_atr:
            return round(max(0.0, (atr_percent / min_atr) * 60), 2)
        if atr_percent > max_atr:
            over = atr_percent - max_atr
            return round(max(0.0, 100 - over * 15), 2)
        return 100.0

    def calculate(
        self,
        strategy_score: float,
        trend_score: float,
        pattern_score: float,
        volume_spike: bool,
        atr: float = 0.0,
        close: float = 0.0
    ) -> float:
        """0–100 के बीच confidence score लौटाए।"""
        # Clamp inputs
        strategy_score = max(0, min(100, strategy_score))
        trend_score = max(0, min(100, trend_score))
        pattern_score = max(0, min(100, pattern_score))

        volume_boost = 20.0 if volume_spike else 0.0
        atr_percent = (atr / close * 100) if close > 0 else 0.0
        atr_quality = self._compute_atr_quality(atr_percent)

        weighted = (
            self.weight_strategy * strategy_score +
            self.weight_trend * trend_score +
            self.weight_pattern * pattern_score +
            self.weight_volume * volume_boost +
            self.weight_atr * atr_quality
        )

        confidence = max(0, min(100, round(weighted, 2)))
        logger.debug(f"Confidence: {confidence} (S={strategy_score}, T={trend_score}, "
                     f"P={pattern_score}, V={volume_boost}, A={atr_quality})")
        return confidence