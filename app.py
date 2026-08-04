import streamlit as st
import pandas as pd
import traceback
import sys
import os

# Add engines folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

from config import *
from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.pattern_engine import PatternEngine
from engines.strategy_engine import StrategyEngine
from engines.confidence_engine import ConfidenceEngine
from engines.risk_engine import RiskEngine
from engines.alert_engine import AlertEngine
from engines.dashboard import DashboardEngine
from engines.trade_logger import TradeLogger
from engines.performance_analyzer import PerformanceAnalyzer
from engines.backtest_engine import BacktestEngine
from engines.backtest_analyzer import BacktestAnalyzer


class StockScanner:

    def __init__(self):
        self.data_engine = DataEngine()
        self.indicator_engine = IndicatorEngine()
        self.pattern_engine = PatternEngine()
        self.strategy_engine = StrategyEngine()
        self.confidence_engine = ConfidenceEngine()
        self.risk_engine = RiskEngine()
        self.alert_engine = AlertEngine()
        self.dashboard = DashboardEngine()
        self.logger = TradeLogger()
        self.performance = PerformanceAnalyzer()
        self.backtest = BacktestEngine()
        self.backtest_analyzer = BacktestAnalyzer()
        self.capital = STARTING_CAPITAL
        self.results = []

    def load_market(self):
        """Load market data for all symbols."""
        st.write("📥 Loading Market Data...")
        self.market_data = self.data_engine.scan_ready_data()
        st.write(f"✅ Loaded {len(self.market_data)} Symbols")
        return self.market_data

    def process_symbol(self, symbol: str, data: pd.DataFrame) -> None:
        """Process a single symbol through the entire pipeline."""
        if data.empty:
            return

        data, indicator_score = self.indicator_engine.process(data)
        data, pattern_score = self.pattern_engine.process(data)
        strategy_result = self.strategy_engine.evaluate(data)

        confidence = self.confidence_engine.calculate(
            strategy_score=strategy_result["strategy_score"],
            trend_score=data["TrendScore"].iloc[-1] if "TrendScore" in data.columns else 0,
            pattern_score=pattern_score,
            volume_spike=data["VOL_SPIKE"].iloc[-1] if "VOL_SPIKE" in data.columns else False,
            atr=data["ATR"].iloc[-1] if "ATR" in data.columns else 0
        )

        trade = self.risk_engine.trade_plan(data, self.capital)
        if not trade:
            return

        self.results.append({
            "Symbol": symbol,
            "Signal": strategy_result["signal"],
            "Confidence": confidence,
            "StrategyScore": strategy_result["strategy_score"],
            "TriggeredStrategies": ", ".join(strategy_result["triggered_strategies"]),
            "PatternScore": pattern_score,
            "SL": trade["StopLoss"],
            "Qty": trade["Quantity"],
            "CurrentPrice": trade["CurrentPrice"],
            "Entry": trade["Entry"],
            "Target1": trade["Target1"],
            "Target2": trade["Target2"],
            "Target3": trade["Target3"],
            "RR": trade["RR"],
        })

    def run(self):
        """Run the complete scanner pipeline."""
        self.load_market()

        progress_bar = st.progress(0)
        total = len(self.market_data)
        
        for idx, (symbol, data) in enumerate(self.market_data.items()):
            try:
                self.process_symbol(symbol, data)
                progress_bar.progress((idx + 1) / total)
            except Exception as e:
                st.write(f"❌ Error in: {symbol}")
                st.code(traceback.format_exc())

        self.results = pd.DataFrame(self.results)
        
        # Backtest
        backtest_results = {}
        progress_bar = st.progress(0)
        for idx, (symbol, data) in enumerate(self.market_data.items()):
            if data.empty:
                continue
            try:
                bt_data, _ = self.indicator_engine.process(data.copy())
                raw_report = self.backtest.run(bt_data)
                analysis = self.backtest_analyzer.analyze(raw_report)
                backtest_results[symbol] = analysis
                progress_bar.progress((idx + 1) / total)
            except Exception as e:
                st.write(f"❌ Backtest Error: {symbol}")
                st.code(traceback.format_exc())

        self.backtest_results = backtest_results
        return self.results

    def send_alerts(self):
        """Send alerts for BUY and SELL signals."""
        if self.results.empty:
            return
        for _, row in self.results.iterrows():
            if row["Signal"] == "WATCH":
                continue
            self.alert_engine.process(
                signal=row["Signal"],
                symbol=row["Symbol"],
                price=row["Entry"]
            )

    def save_logs(self):
        """Save all trades to log."""
        if self.results.empty:
            return
        for _, row in self.results.iterrows():
            self.logger.save_trade(
                symbol=row["Symbol"],
                signal=row["Signal"],
                entry=row["Entry"],
                exit_price=0,
                sl=row["SL"],
                target=row["Target1"],
                qty=row["Qty"],
                pnl=0,
                pnl_percent=0,
                reason="Signal Generated",
                confidence=row["Confidence"],
                ema=0,
                macd=0,
                rsi=0,
                pattern=row["PatternScore"],
                trend=0
            )


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="AI Stock Scanner V1.4", layout="wide")
    st.title("🤖 AI Stock Scanner V1.4")
    st.markdown("---")

    scanner = StockScanner()
    results = scanner.run()

    # Display scanner results
    st.subheader("📊 Scanner Results")
    if not results.empty:
        st.dataframe(results, use_container_width=True)
        
        # Signal counts
        col1, col2, col3 = st.columns(3)
        col1.metric("BUY Signals", len(results[results["Signal"] == "BUY"]))
        col2.metric("SELL Signals", len(results[results["Signal"] == "SELL"]))
        col3.metric("WATCH Signals", len(results[results["Signal"] == "WATCH"]))
    else:
        st.warning("⚠️ No signals found.")

    st.markdown("---")
    
    # Overall Backtest Dashboard
    st.subheader("📈 Overall Backtest Dashboard")

    overall_stats = scanner.dashboard.overall_stats(scanner.backtest_results)

    if overall_stats["Total Trades"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", int(overall_stats["Total Trades"]))
        col2.metric("Wins", int(overall_stats["Wins"]))
        col3.metric("Losses", int(overall_stats["Losses"]))
        col4.metric("Win Rate", f"{overall_stats['Win Rate']}%")

        col5, col6, col7 = st.columns(3)
        col5.metric("Total P&L", f"₹{round(overall_stats['Total PnL'], 2)}")
        col6.metric("Profit Factor", round(overall_stats["Profit Factor"], 2))
        col7.metric("AI Score", int(overall_stats["AI Score"]))
    else:
        st.info("ℹ️ No backtest results available.")

    st.markdown("---")
    
    # Stock Ranking
    st.subheader("🏆 Stock Ranking")
    ranking_df = scanner.dashboard.ranking_table(scanner.backtest_results)

    if not ranking_df.empty:
        st.dataframe(ranking_df, use_container_width=True)
    else:
        st.info("ℹ️ No ranking data available.")

    st.markdown("---")
    
    # Backtest Summary
    st.subheader("📋 Backtest Summary")
    
    # Show only symbols with trades
    active_symbols = {s: r for s, r in scanner.backtest_results.items() 
                     if isinstance(r, dict) and r.get("Total Trades", 0) > 0}
    
    if active_symbols:
        for symbol, report in active_symbols.items():
            with st.expander(f"📊 {symbol} - {report.get('Total Trades', 0)} Trades"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Win Rate", f"{report.get('Win Rate', 0)}%")
                col2.metric("Total P&L", f"₹{report.get('Total PnL', 0)}")
                col3.metric("AI Score", report.get('AI Score', 0))
                
                st.write({
                    "Total Trades": report.get("Total Trades", 0),
                    "Wins": report.get("Wins", 0),
                    "Losses": report.get("Losses", 0),
                    "Profit Factor": report.get("Profit Factor", 0),
                    "Max Drawdown": report.get("Max Drawdown", 0),
                })
    else:
        st.info("ℹ️ No active trades to display.")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main()