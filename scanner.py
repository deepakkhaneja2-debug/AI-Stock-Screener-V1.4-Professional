import logging
import traceback
from typing import Dict, Any

import pandas as pd

import config
from config import STARTING_CAPITAL
from data_engine import DataEngine
from indicator_engine import IndicatorEngine
from pattern_engine import PatternEngine
from strategy_engine import StrategyEngine
from confidence_engine import ConfidenceEngine
from risk_engine import RiskEngine
from backtest_engine import BacktestEngine
from backtest_analyzer import BacktestAnalyzer
from dashboard import DashboardEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StockScanner:
    """
    Main scanner pipeline used by app.py.
    Loads market data, runs indicators/patterns/strategy/risk,
    then runs backtest and exposes backtest_results.
    """

    def __init__(self):
        self.data_engine = DataEngine()
        self.indicator_engine = IndicatorEngine()
        self.pattern_engine = PatternEngine()
        self.strategy_engine = StrategyEngine()
        self.confidence_engine = ConfidenceEngine()
        self.risk_engine = RiskEngine()
        self.backtest = BacktestEngine()
        self.backtest_analyzer = BacktestAnalyzer()
        self.dashboard = DashboardEngine()

        self.capital = STARTING_CAPITAL
        self.results = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.backtest_results: Dict[str, Any] = {}
        self.backtest_debug: Dict[str, Any] = {}
        self.market_regime: pd.Series = pd.Series(dtype=object)

    def load_market(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all symbols."""
        logger.info("Loading market data...")
        self.market_data = self.data_engine.scan_ready_data() or {}
        logger.info("Loaded %s symbols", len(self.market_data))
        return self.market_data

    def compute_market_regime(self) -> pd.Series:
        """
        Build a per-day NIFTY market regime series (STEP 6):
        NIFTY Close > EMA20 > EMA50  -> BULLISH
        NIFTY Close < EMA20 < EMA50  -> BEARISH
        otherwise                   -> NEUTRAL

        Returned series is aligned to NIFTY's own date index and uses
        only that day's (and earlier) values - no look-ahead.
        """
        try:
            market_indices = self.data_engine.get_market_index()
        except Exception:
            logger.error("Market index fetch failed:\n%s", traceback.format_exc())
            return pd.Series(dtype=object)

        nifty = market_indices.get("NIFTY")
        if nifty is None or nifty.empty:
            return pd.Series(dtype=object)

        try:
            nifty_ind, _ = self.indicator_engine.process(nifty.copy())
            close = nifty_ind["Close"]
            ema20 = nifty_ind["EMA20"]
            ema50 = nifty_ind["EMA50"]

            regime = pd.Series("NEUTRAL", index=nifty_ind.index)
            bullish = (close > ema20) & (ema20 > ema50)
            bearish = (close < ema20) & (ema20 < ema50)
            regime[bullish] = "BULLISH"
            regime[bearish] = "BEARISH"

            self.market_regime = regime
            return regime
        except Exception:
            logger.error("Market regime computation failed:\n%s", traceback.format_exc())
            return pd.Series(dtype=object)

    def process_symbol(self, symbol: str, data: pd.DataFrame) -> None:
        """Process one symbol through the scanner pipeline."""
        if data is None or data.empty:
            return

        # Indicators
        data, indicator_score = self.indicator_engine.process(data)

        # Patterns
        data, pattern_score = self.pattern_engine.process(data)

        # Strategy
        strategy_result = self.strategy_engine.evaluate(data)

        strategy_score = strategy_result.get("strategy_score", 0)
        signal = strategy_result.get("signal", "WATCH")
        triggered = strategy_result.get("triggered_strategies", [])

        # AI Score / Confidence: use strategy_engine's own values directly
        # (it already computes both from the SAME inputs that produced the
        # signal). Previously scanner.py discarded these and recomputed a
        # second, independent "Confidence" via ConfidenceEngine using a
        # different weighting - two uncoordinated numbers for the same
        # concept, which is exactly the "AI Score vs Confidence
        # contradiction" risk. Using one source of truth removes that.
        ai_score = strategy_result.get("ai_score", 0)
        confidence = strategy_result.get("confidence_score", 0)

        # Risk / trade plan
        trade = self.risk_engine.trade_plan(data, self.capital)
        if not trade:
            return

        self.results.append({
            "Symbol": symbol,
            "Signal": signal,
            "AI Score": ai_score,
            "Confidence": confidence,
            "StrategyScore": strategy_score,
            "TriggeredStrategies": ", ".join(map(str, triggered)),
            "PatternScore": pattern_score,
            "TrendScore": data["TrendScore"].iloc[-1] if "TrendScore" in data.columns and not data.empty else 0,
            "SL": trade.get("StopLoss", 0),
            "Qty": trade.get("Quantity", 0),
            "CurrentPrice": trade.get("CurrentPrice", 0),
            "Entry": trade.get("Entry", 0),
            "Target1": trade.get("Target1", 0),
            "Target2": trade.get("Target2", 0),
            "Target3": trade.get("Target3", 0),
            "RR": trade.get("RR", 0),
        })

    def run(self) -> pd.DataFrame:
        """Run scanner and backtest for all symbols."""
        self.results = []
        self.backtest_results = {}
        self.backtest_debug = {}

        self.load_market()

        # Market regime (STEP 6) - computed once, reused for every
        # symbol's backtest so all stocks see the same NIFTY context.
        market_regime = self.compute_market_regime()
        if market_regime is None or market_regime.empty:
            logger.warning(
                "Market regime unavailable - backtests will run with "
                "NEUTRAL regime (market filter effectively disabled)."
            )

        # Live scan
        for symbol, data in self.market_data.items():
            try:
                self.process_symbol(symbol, data)
            except Exception:
                logger.error("Scanner error for %s:\n%s", symbol, traceback.format_exc())

        self.results = pd.DataFrame(self.results)

        # Backtest
        for symbol, data in self.market_data.items():
            if data is None or data.empty:
                continue

            try:
                bt_data, _ = self.indicator_engine.process(data.copy())
                bt_data, _ = self.pattern_engine.process(bt_data)
                raw_report = self.backtest.run(
                    bt_data,
                    market_regime=market_regime,
                    symbol=symbol
                )
                analysis = self.backtest_analyzer.analyze(raw_report)
                self.backtest_results[symbol] = analysis
                self.backtest_debug[symbol] = analysis.get("Debug", {})

                if getattr(config, "BACKTEST_DEBUG", False):
                    logger.info("Backtest diagnostics for %s: %s", symbol, analysis.get("Debug", {}))

            except Exception:
                logger.error("Backtest error for %s:\n%s", symbol, traceback.format_exc())

        return self.results


if __name__ == "__main__":
    scanner = StockScanner()
    results = scanner.run()
    print(results.to_string(index=False) if not results.empty else "No results")