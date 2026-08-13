import logging
from typing import Dict, Any, List, Optional

import pandas as pd

import config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BacktestEngine:
    """
    AI Stock Scanner V1.4
    Complete swing backtest engine.

    Features:
    - EMA20 / EMA50 trend
    - MACD confirmation
    - RSI filter
    - VWAP filter
    - ATR based entry
    - ATR based stop loss
    - Risk based position sizing
    - Target 1 / Target 2 / Target 3
    - Break-even management
    - Trailing stop
    - Slippage
    - Brokerage
    - Equity curve
    - Trade statistics
    """

    def __init__(self):

        self.lookahead_days = getattr(
            config,
            "BACKTEST_LOOKAHEAD",
            30
        )

        self.entry_atr_buffer = getattr(
            config,
            "ENTRY_ATR_BUFFER",
            0.25
        )

        self.stop_atr_multiplier = getattr(
            config,
            "STOP_ATR_MULTIPLIER",
            1.5
        )

        self.target1_r = getattr(
            config,
            "TARGET1_R",
            1.5
        )

        self.target2_r = getattr(
            config,
            "TARGET2_R",
            2.5
        )

        self.target3_r = getattr(
            config,
            "TARGET3_R",
            4.0
        )

        self.use_break_even = getattr(
            config,
            "BREAK_EVEN_AT_TARGET1",
            True
        )

        self.trailing_stop_atr = getattr(
            config,
            "TRAILING_STOP_ATR",
            2.0
        )

        self.min_trades_for_ranking = getattr(
            config,
            "MIN_TRADES_FOR_RANKING",
            5
        )

        self.risk_per_trade = getattr(
            config,
            "RISK_PER_TRADE",
            1.0
        )

        self.initial_capital = getattr(
            config,
            "STARTING_CAPITAL",
            100000
        )

        self.slippage = getattr(
            config,
            "SLIPPAGE",
            0.001
        )

        self.brokerage = getattr(
            config,
            "BROKERAGE_PER_TRADE",
            20
        )

        self.enable_transaction_cost = getattr(
            config,
            "ENABLE_TRANSACTION_COST",
            True
        )

        self.brokerage_type = getattr(
            config,
            "BROKERAGE_TYPE",
            "fixed"
        )

        # --- new: score-based entry model (STEP 5) ---
        self.entry_window = getattr(config, "BACKTEST_ENTRY_WINDOW", 15)
        self.min_signal_score = getattr(config, "MIN_SIGNAL_SCORE", 60)
        self.w_trend = getattr(config, "SCORE_WEIGHT_TREND", 25)
        self.w_momentum = getattr(config, "SCORE_WEIGHT_MOMENTUM", 20)
        self.w_volume = getattr(config, "SCORE_WEIGHT_VOLUME", 15)
        self.w_pattern = getattr(config, "SCORE_WEIGHT_PATTERN", 15)
        self.w_market = getattr(config, "SCORE_WEIGHT_MARKET", 15)
        self.w_volatility = getattr(config, "SCORE_WEIGHT_VOLATILITY", 10)

        # --- new: ATR% volatility filter (STEP 7) ---
        self.min_atr_percent = getattr(config, "MIN_ATR_PERCENT", 0.5)
        self.max_atr_percent = getattr(config, "MAX_ATR_PERCENT", 6.0)

        # --- new: volume filter ---
        self.min_volume_ratio = getattr(config, "MIN_VOLUME_RATIO", 0.8)

        # --- new: market regime filter (STEP 6) ---
        self.use_market_regime_filter = getattr(config, "USE_MARKET_REGIME_FILTER", True)
        self.market_bullish_bonus = getattr(config, "MARKET_BULLISH_SCORE_BONUS", 10)
        self.market_bearish_penalty = getattr(config, "MARKET_BEARISH_SCORE_PENALTY", 15)
        self.market_neutral_penalty = getattr(config, "MARKET_NEUTRAL_MIN_SCORE_BONUS", 5)

        self.debug_enabled = getattr(config, "BACKTEST_DEBUG", False)

    # ============================================================
    # SCORE A SINGLE CANDLE (STEP 5 / 6 / 7)
    # ============================================================
    def _score_buy_setup(self, row, prev_row, regime: str) -> Dict[str, Any]:
        """
        Score a candle 0-100 for BUY quality using weighted components
        instead of a rigid all-or-nothing AND gate. Uses ONLY values
        already present on this candle (or earlier) - no look-ahead.
        Returns dict with total score and component breakdown so the
        result can be logged for diagnostics (STEP 19).
        """
        ema20 = float(row.get("EMA20", 0) or 0)
        ema50 = float(row.get("EMA50", 0) or 0)
        rsi = float(row.get("RSI", 50) or 50)
        macd = float(row.get("MACD", 0) or 0)
        macd_signal = float(row.get("MACD_SIGNAL", 0) or 0)
        close = float(row.get("Close", 0) or 0)
        atr = float(row.get("ATR", 0) or 0)
        vwap = float(row.get("VWAP", close) or close)
        volume = float(row.get("Volume", 0) or 0)
        vol_ma20 = float(row.get("VOL_MA", row.get("VOL_MA20", 0)) or 0)
        vol_spike = bool(row.get("VOL_SPIKE", False))
        bullish_pattern = bool(row.get("BULLISH_ENGULFING", False) or row.get("HAMMER", False))

        # ---- Trend component ----
        trend_pts = 0.0
        if ema20 > ema50:
            trend_pts += 0.6
        if close > ema20:
            trend_pts += 0.4
        trend_score = trend_pts * self.w_trend

        # ---- Momentum component ----
        momentum_pts = 0.0
        if macd > macd_signal:
            momentum_pts += 0.5
        if 50 <= rsi <= 70:
            momentum_pts += 0.5
        elif rsi > 70 or rsi < 45:
            momentum_pts -= 0.2
        momentum_pts = max(0.0, min(1.0, momentum_pts))
        momentum_score = momentum_pts * self.w_momentum

        # ---- Volume component ----
        volume_pts = 0.0
        if vol_ma20 > 0 and volume > 0:
            ratio = volume / vol_ma20
            if ratio >= self.min_volume_ratio:
                volume_pts += 0.6
            if vol_spike:
                volume_pts += 0.4
        else:
            # No volume data available - stay neutral instead of
            # zeroing the whole trade out.
            volume_pts = 0.5
        volume_pts = min(1.0, volume_pts)
        volume_score = volume_pts * self.w_volume

        # ---- Pattern component ----
        pattern_pts = 0.0
        if bullish_pattern:
            pattern_pts += 0.6
        if prev_row is not None:
            prev_close = float(prev_row.get("Close", close) or close)
            if close > prev_close:
                pattern_pts += 0.4
        pattern_pts = min(1.0, pattern_pts)
        pattern_score = pattern_pts * self.w_pattern

        # ---- Market regime component (STEP 6) ----
        market_pts = 0.5
        if self.use_market_regime_filter:
            if regime == "BULLISH":
                market_pts = 1.0
            elif regime == "BEARISH":
                market_pts = 0.15
            else:
                market_pts = 0.55
        market_score = market_pts * self.w_market

        # ---- Volatility component (ATR% filter, STEP 7) ----
        atr_percent = (atr / close * 100) if close > 0 else 0
        volatility_pts = 0.0
        atr_ok = self.min_atr_percent <= atr_percent <= self.max_atr_percent
        if atr_ok:
            volatility_pts = 1.0
        volatility_score = volatility_pts * self.w_volatility

        total = (
            trend_score + momentum_score + volume_score
            + pattern_score + market_score + volatility_score
        )

        return {
            "total": round(total, 2),
            "trend": round(trend_score, 2),
            "momentum": round(momentum_score, 2),
            "volume": round(volume_score, 2),
            "pattern": round(pattern_score, 2),
            "market": round(market_score, 2),
            "volatility": round(volatility_score, 2),
            "atr_percent": round(atr_percent, 3),
            "atr_ok": atr_ok,
        }



    # ============================================================
    # RUN BACKTEST
    # ============================================================

    def run(
        self,
        data: pd.DataFrame,
        market_regime: Optional[pd.Series] = None,
        symbol: str = ""
    ) -> Dict[str, Any]:
        """
        Run complete backtest.

        market_regime: optional pd.Series aligned to `data`'s index,
        with values in {"BULLISH", "NEUTRAL", "BEARISH"} representing
        the NIFTY market regime on that date (STEP 6). Only regime
        values at or before the signal candle are ever used - no
        look-ahead.
        """

        debug = {
            "symbol": symbol,
            "rows": 0,
            "valid_rows": 0,
            "signals_scored": 0,
            "signals_passed_score": 0,
            "entry_candidates": 0,
            "entries_executed": 0,
            "rejected_atr_filter": 0,
            "rejected_market_filter": 0,
            "rejected_entry_not_filled": 0,
            "rejected_risk_too_small": 0,
            "trades_closed": 0,
        }

        if data is None or data.empty:

            logger.warning(
                "No data provided for backtest"
            )

            return self._summary([], debug=debug)

        debug["rows"] = len(data)


        data = data.copy()


        # --------------------------------------------------------
        # REQUIRED COLUMNS
        # --------------------------------------------------------

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "EMA20",
            "EMA50",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "ATR",
        ]


        missing = [
            col
            for col in required
            if col not in data.columns
        ]


        if missing:

            logger.warning(
                "Missing backtest columns: %s",
                missing
            )

            return self._summary([], debug=debug)


        # --------------------------------------------------------
        # VWAP OPTIONAL
        # --------------------------------------------------------

        data = data.dropna(
            subset=required
        )


        if "VWAP" in data.columns:

            data = data.dropna(
                subset=["VWAP"]
            )

        debug["valid_rows"] = len(data)

        # Need enough history
        if len(data) < 61:

            logger.warning(
                "Insufficient data for backtest: %s rows",
                len(data)
            )

            return self._summary([], debug=debug)


        data = data.sort_index()

        # Align market regime to the (possibly reduced) data index.
        # Forward-fill only - never bfill, since that would leak a
        # future regime value backwards onto earlier candles.
        if market_regime is not None:
            market_regime = market_regime.reindex(data.index).ffill().fillna("NEUTRAL")
        else:
            market_regime = pd.Series("NEUTRAL", index=data.index)


        # --------------------------------------------------------
        # BACKTEST VARIABLES
        # --------------------------------------------------------

        results: List[Dict[str, Any]] = []

        equity_curve: List[Dict[str, Any]] = []

        cumulative_pnl = 0.0

        trade_counter = 0

        next_available_index = 60


        # ========================================================
        # MAIN LOOP
        # ========================================================

        for i in range(
            60,
            len(data) - 1
        ):

            if i < next_available_index:
                continue


            row = data.iloc[i]


            # ----------------------------------------------------
            # READ INDICATORS
            # ----------------------------------------------------

            try:

                ema20 = float(
                    row["EMA20"]
                )

                ema50 = float(
                    row["EMA50"]
                )

                rsi = float(
                    row["RSI"]
                )

                macd = float(
                    row["MACD"]
                )

                macd_signal = float(
                    row["MACD_SIGNAL"]
                )

                close = float(
                    row["Close"]
                )

                atr = float(
                    row["ATR"]
                )

                if "VWAP" in data.columns:

                    vwap = float(
                        row["VWAP"]
                    )

                else:

                    vwap = close


            except (
                TypeError,
                ValueError
            ) as exc:

                logger.debug(
                    "Indicator read error at %s: %s",
                    i,
                    exc
                )

                continue


            # ----------------------------------------------------
            # VALIDATE VALUES
            # ----------------------------------------------------

            if (
                atr <= 0
                or close <= 0
                or pd.isna(atr)
                or pd.isna(close)
            ):

                continue


            # ====================================================
            # BUY SETUP - WEIGHTED SCORE MODEL (STEP 5/6/7)
            # ====================================================

            debug["signals_scored"] += 1

            regime = str(market_regime.iloc[i]) if len(market_regime) > i else "NEUTRAL"
            prev_row = data.iloc[i - 1] if i > 0 else None

            score_detail = self._score_buy_setup(row, prev_row, regime)

            if not score_detail["atr_ok"]:
                debug["rejected_atr_filter"] += 1
                continue

            if self.use_market_regime_filter and regime == "BEARISH" and score_detail["total"] < (self.min_signal_score + self.market_bearish_penalty):
                debug["rejected_market_filter"] += 1
                continue

            buy_setup = score_detail["total"] >= self.min_signal_score

            if not buy_setup:
                continue

            debug["signals_passed_score"] += 1


            # ====================================================
            # THEORETICAL TRADE PLAN
            # ====================================================

            theoretical_entry = round(
                close
                + (
                    atr
                    * self.entry_atr_buffer
                ),
                2
            )


            theoretical_stoploss = round(
                theoretical_entry
                - (
                    atr
                    * self.stop_atr_multiplier
                ),
                2
            )

            # Volatility-aware stop (STEP 8): also consider the
            # recent swing low and use the SAFER (higher / less
            # negative-risk) of the two stops, never the riskier one.
            try:
                swing_low = float(
                    data.iloc[max(0, i - 10):i + 1]["Low"].min()
                )
                swing_stop = round(swing_low - (atr * 0.1), 2)
                if swing_stop > 0:
                    theoretical_stoploss = max(theoretical_stoploss, swing_stop) \
                        if swing_stop < theoretical_entry else theoretical_stoploss
                    # Use whichever stop is closer to entry (safer / smaller risk)
                    # but never allow a stop above/at entry.
                    theoretical_stoploss = min(theoretical_stoploss, theoretical_entry - 0.01)
            except (ValueError, KeyError) as exc:
                logger.debug(
                    "Swing-low stop fallback triggered for %s at candle %s (%s): %s "
                    "- using ATR-only stop instead",
                    symbol,
                    i,
                    row.name,
                    exc
                )


            theoretical_risk = round(
                theoretical_entry
                - theoretical_stoploss,
                2
            )


            if theoretical_risk <= 0:
                continue


            theoretical_target1 = round(
                theoretical_entry
                + (
                    theoretical_risk
                    * self.target1_r
                ),
                2
            )


            theoretical_target2 = round(
                theoretical_entry
                + (
                    theoretical_risk
                    * self.target2_r
                ),
                2
            )


            theoretical_target3 = round(
                theoretical_entry
                + (
                    theoretical_risk
                    * self.target3_r
                ),
                2
            )


            # ====================================================
            # FIND ACTUAL ENTRY
            # ====================================================

            entry_index: Optional[int] = None

            actual_entry_price: Optional[float] = None

            debug["entry_candidates"] += 1

            entry_end = min(
                i + self.entry_window,
                len(data)
            )


            for j in range(
                i + 1,
                entry_end
            ):

                try:

                    candle_open = float(
                        data.iloc[j]["Open"]
                    )

                    candle_high = float(
                        data.iloc[j]["High"]
                    )


                except (
                    TypeError,
                    ValueError
                ):

                    continue


                if (
                    candle_high
                    >= theoretical_entry
                ):

                    if (
                        candle_open
                        > theoretical_entry
                    ):

                        actual_entry_price = (
                            candle_open
                        )

                    else:

                        actual_entry_price = (
                            theoretical_entry
                        )


                    entry_index = j

                    break


            if (
                entry_index is None
                or actual_entry_price is None
            ):

                debug["rejected_entry_not_filled"] += 1
                continue


            # ====================================================
            # ACTUAL TRADE LEVELS
            # ====================================================

            stoploss = round(
                actual_entry_price
                - (
                    atr
                    * self.stop_atr_multiplier
                ),
                2
            )


            actual_risk = round(
                actual_entry_price
                - stoploss,
                2
            )


            if actual_risk <= 0:
                continue


            actual_target1 = round(
                actual_entry_price
                + (
                    actual_risk
                    * self.target1_r
                ),
                2
            )


            actual_target2 = round(
                actual_entry_price
                + (
                    actual_risk
                    * self.target2_r
                ),
                2
            )


            actual_target3 = round(
                actual_entry_price
                + (
                    actual_risk
                    * self.target3_r
                ),
                2
            )


            # ====================================================
            # POSITION SIZE
            # ====================================================

            capital = (
                self.initial_capital
                + cumulative_pnl
            )


            risk_amount = (
                capital
                * (
                    self.risk_per_trade
                    / 100
                )
            )


            quantity = int(
                risk_amount
                / max(
                    actual_risk,
                    0.01
                )
            )

            if quantity <= 0:
                # Risk-per-trade too small relative to this stock's
                # stop distance to buy even 1 share without exceeding
                # RISK_PER_TRADE. Forcing a minimum of 1 share here
                # (the previous behaviour) silently let realized risk
                # exceed the configured limit - most likely to happen
                # for expensive/high-ATR stocks and, worse, becomes
                # MORE likely during a drawdown since capital shrinks.
                # Skip the trade instead of understating its true risk.
                debug["rejected_risk_too_small"] += 1
                continue


            trade_counter += 1

            debug["entries_executed"] += 1

            logger.info(
                "Trade %s entry | Entry %.2f | "
                "SL %.2f | Qty %s",
                trade_counter,
                actual_entry_price,
                stoploss,
                quantity
            )


            # ====================================================
            # TRADE OBJECT
            # ====================================================

            expiry = min(
                entry_index
                + self.lookahead_days,
                len(data)
            )


            trade = {

                "TradeID":
                    trade_counter,

                "Symbol":
                    symbol,

                "Date":
                    data.iloc[
                        entry_index
                    ].name,

                "EntryDate":
                    data.iloc[
                        entry_index
                    ].name,

                "SignalDate":
                    row.name,

                "Entry":
                    round(
                        actual_entry_price,
                        2
                    ),

                "StopLoss":
                    stoploss,

                "Risk":
                    actual_risk,

                "Target1":
                    actual_target1,

                "Target2":
                    actual_target2,

                "Target3":
                    actual_target3,

                "Expiry":
                    expiry,

                "Status":
                    "OPEN",

                "EntryIndex":
                    entry_index,

                "CurrentStop":
                    stoploss,

                "HighestPrice":
                    actual_entry_price,

                "LowestPrice":
                    actual_entry_price,

                "TargetHit":
                    None,

                "ExitPrice":
                    None,

                "ExitDate":
                    None,

                "ExitReason":
                    None,

                "HoldingDays":
                    0,

                "PnL":
                    0.0,

                "PnLPercent":
                    0.0,

                "RMultiple":
                    0.0,

                "TradeReason":
                    (
                        f"Score {score_detail['total']:.0f}/100 "
                        f"(Trend {score_detail['trend']:.0f} "
                        f"Momentum {score_detail['momentum']:.0f} "
                        f"Volume {score_detail['volume']:.0f} "
                        f"Pattern {score_detail['pattern']:.0f} "
                        f"Market {score_detail['market']:.0f} "
                        f"Volatility {score_detail['volatility']:.0f}) "
                        f"| Regime {regime}"
                    ),

                "RR":
                    round(
                        (
                            actual_target2
                            - actual_entry_price
                        )
                        / actual_risk,
                        2
                    ),

                "Quantity":
                    quantity,

                "CapitalUsed":
                    capital,

                "RiskAmount":
                    risk_amount,

                "EntrySlippage":
                    0.0,

                "ExitSlippage":
                    0.0,

                "Brokerage":
                    0.0,

                "GrossPnL":
                    0.0,

                "NetPnL":
                    0.0,

                "RunningEquity":
                    capital,

            }


            trade_closed = False

            break_even_triggered = False


            # ====================================================
            # TRADE MANAGEMENT
            # ====================================================

            for j in range(
                entry_index,
                expiry
            ):

                candle = data.iloc[j]


                try:

                    low = float(
                        candle["Low"]
                    )

                    high = float(
                        candle["High"]
                    )

                    candle_open = float(
                        candle["Open"]
                    )

                    candle_close = float(
                        candle["Close"]
                    )


                    if "ATR" in data.columns:

                        current_atr = float(
                            candle["ATR"]
                        )

                    else:

                        current_atr = (
                            trade["Risk"]
                            / 1.5
                        )


                except (
                    TypeError,
                    ValueError
                ):

                    continue


                if current_atr <= 0:

                    current_atr = (
                        trade["Risk"]
                        / 1.5
                    )


                trade["HoldingDays"] = (
                    j
                    - trade["EntryIndex"]
                    + 1
                )


                trade["HighestPrice"] = max(
                    trade["HighestPrice"],
                    high
                )


                trade["LowestPrice"] = min(
                    trade["LowestPrice"],
                    low
                )


                # =================================================
                # TARGET 1 / BREAK EVEN / TRAILING
                # =================================================

                if high >= trade["Target1"]:

                    if self.use_break_even:

                        trade["CurrentStop"] = max(
                            trade["CurrentStop"],
                            trade["Entry"]
                        )

                        break_even_triggered = True


                    else:

                        self._close_trade(
                            trade,
                            j,
                            "TARGET1",
                            data
                        )

                        trade_closed = True

                        break

                
                # =================================================
                # TARGET 2
                # =================================================

                if (
                    not trade_closed
                    and high >= trade["Target2"]
                ):

                    self._close_trade(
                        trade,
                        j,
                        "TARGET2",
                        data
                    )

                    trade_closed = True

                    break


                # =================================================
                # TARGET 3
                # =================================================

                if (
                    not trade_closed
                    and high >= trade["Target3"]
                ):

                    self._close_trade(
                        trade,
                        j,
                        "TARGET3",
                        data
                    )

                    trade_closed = True

                    break


                # =================================================
                # TRAILING STOP
                # =================================================

                if (
                    not trade_closed
                    and break_even_triggered
                ):

                    new_stop = (
                        high
                        - (
                            current_atr
                            * self.trailing_stop_atr
                        )
                    )


                    new_stop = min(
                        new_stop,
                        trade["Target3"]
                    )


                    new_stop = max(
                        new_stop,
                        trade["CurrentStop"]
                    )


                    trade["CurrentStop"] = round(
                        new_stop,
                        2
                    )


                # =================================================
                # STOP LOSS
                # =================================================

                if (
                    not trade_closed
                    and low
                    <= trade["CurrentStop"]
                ):

                    if (
                        candle_open
                        < trade["CurrentStop"]
                    ):

                        exit_price = (
                            candle_open
                        )

                    else:

                        exit_price = (
                            trade["CurrentStop"]
                        )


                    trade["ExitPrice"] = (
                        exit_price
                    )


                    if break_even_triggered:

                        reason = "BREAK_EVEN"

                    else:

                        reason = "STOP_LOSS"


                    self._close_trade(
                        trade,
                        j,
                        reason,
                        data
                    )


                    trade_closed = True

                    break


            # ====================================================
            # TIME EXIT
            # ====================================================

            if (
                not trade_closed
                and trade["Status"] == "OPEN"
            ):

                last_idx = min(
                    expiry - 1,
                    len(data) - 1
                )


                trade["ExitPrice"] = float(
                    data.iloc[
                        last_idx
                    ]["Close"]
                )


                self._close_trade(
                    trade,
                    last_idx,
                    "TIME_EXIT",
                    data
                )


                trade_closed = True


            # ====================================================
            # SAVE TRADE
            # ====================================================

            if trade["Status"] != "OPEN":

                cumulative_pnl += (
                    trade["NetPnL"]
                )


                trade["RunningEquity"] = round(
                    self.initial_capital
                    + cumulative_pnl,
                    2
                )


                results.append(
                    trade
                )

                debug["trades_closed"] += 1

                equity_curve.append({

                    "Date":
                        trade["ExitDate"],

                    "Equity":
                        trade["RunningEquity"],

                    "CumulativePnL":
                        round(
                            cumulative_pnl,
                            2
                        )

                })


                logger.info(
                    "Trade %s closed | %s | "
                    "PnL %.2f | R %.2f",
                    trade["TradeID"],
                    trade["ExitReason"],
                    trade["NetPnL"],
                    trade["RMultiple"]
                )


            # ====================================================
            # PREVENT OVERLAPPING TRADES
            # ====================================================

            next_available_index = max(
                entry_index + 1,
                expiry + 1
            )


        # ========================================================
        # FINAL SUMMARY
        # ========================================================

        return self._summary(
            results,
            equity_curve,
            debug=debug
        )


    # ============================================================
    # CLOSE TRADE
    # ============================================================

    def _close_trade(
        self,
        trade: Dict[str, Any],
        exit_idx: int,
        reason: str,
        data: pd.DataFrame
    ) -> None:
        """
        Close trade and calculate transaction costs.
        """

        if trade["Status"] != "OPEN":
            return


        # --------------------------------------------------------
        # EXIT PRICE
        # --------------------------------------------------------

        if reason == "TARGET3":

            exit_price = (
                trade["Target3"]
            )

        elif reason == "TARGET2":

            exit_price = (
                trade["Target2"]
            )

        elif reason == "TARGET1":

            exit_price = (
                trade["Target1"]
            )

        elif reason in [
            "STOP_LOSS",
            "BREAK_EVEN"
        ]:

            exit_price = trade.get(
                "ExitPrice",
                trade["CurrentStop"]
            )

        else:

            exit_price = trade.get(
                "ExitPrice",
                trade["Entry"]
            )


        entry_price_raw = float(
            trade["Entry"]
        )


        # --------------------------------------------------------
        # TRANSACTION COST
        # --------------------------------------------------------

        if self.enable_transaction_cost:

            entry_slippage = (
                entry_price_raw
                * self.slippage
            )


            exit_slippage = (
                exit_price
                * self.slippage
            )


            # BUY entry => slippage increases price
            entry_price = (
                entry_price_raw
                + entry_slippage
            )


            # SELL exit => slippage decreases price
            exit_price = (
                exit_price
                - exit_slippage
            )


            if (
                str(
                    self.brokerage_type
                ).lower()
                == "percentage"
            ):

                brokerage = (

                    entry_price
                    * trade["Quantity"]
                    * self.brokerage
                    / 100

                ) + (

                    exit_price
                    * trade["Quantity"]
                    * self.brokerage
                    / 100

                )

            else:

                brokerage = (
                    self.brokerage * 2
                )


        else:

            entry_slippage = 0.0

            exit_slippage = 0.0

            entry_price = (
                entry_price_raw
            )

            brokerage = 0.0


        # --------------------------------------------------------
        # P&L
        # --------------------------------------------------------

        quantity = int(
            trade["Quantity"]
        )


        gross_pnl = (
            exit_price
            - entry_price
        ) * quantity


        net_pnl = (
            gross_pnl
            - brokerage
        )


        pnl = round(
            net_pnl,
            2
        )


        # --------------------------------------------------------
        # PERCENT RETURN
        # --------------------------------------------------------

        if (
            entry_price_raw > 0
            and quantity > 0
        ):

            pnl_percent = round(

                (
                    pnl
                    / (
                        entry_price_raw
                        * quantity
                    )
                )
                * 100,

                2
            )

        else:

            pnl_percent = 0.0


        # --------------------------------------------------------
        # R MULTIPLE
        # --------------------------------------------------------

        if trade["Risk"] > 0:

            r_multiple = round(

                (
                    exit_price
                    - entry_price
                )
                / trade["Risk"],

                2
            )

        else:

            r_multiple = 0.0


        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        if pnl > 0:

            status = "WIN"

        elif pnl < 0:

            status = "LOSS"

        else:

            status = "BREAK_EVEN"


        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------

        trade["Status"] = status

        trade["ExitPrice"] = round(
            exit_price,
            2
        )

        trade["ExitDate"] = (
            data.iloc[
                exit_idx
            ].name
        )

        trade["ExitReason"] = reason

        trade["PnL"] = pnl

        trade["PnLPercent"] = (
            pnl_percent
        )

        trade["RMultiple"] = (
            r_multiple
        )

        trade["EntrySlippage"] = round(
            entry_slippage,
            2
        )

        trade["ExitSlippage"] = round(
            exit_slippage,
            2
        )

        trade["Brokerage"] = round(
            brokerage,
            2
        )

        trade["GrossPnL"] = round(
            gross_pnl,
            2
        )

        # Both names are kept for compatibility
        trade["NetPnL"] = pnl


        if reason in [
            "TARGET1",
            "TARGET2",
            "TARGET3"
        ]:

            trade["TargetHit"] = (
                reason
            )


        logger.debug(
            "Trade closed: %s | "
            "Reason=%s | "
            "Gross=%.2f | "
            "Net=%.2f | "
            "R=%.2f",
            status,
            reason,
            gross_pnl,
            pnl,
            r_multiple
        )


    # ============================================================
    # SUMMARY
    # ============================================================

    def _summary(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: Optional[
            List[Dict[str, Any]]
        ] = None,
        debug: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive backtest summary.
        """

        if equity_curve is None:

            equity_curve = []


        if not trades:

            empty = self._empty_summary()
            empty["Debug"] = debug or {}
            return empty


        # --------------------------------------------------------
        # COUNTS
        # --------------------------------------------------------

        wins = sum(
            1
            for t in trades
            if t.get("Status") == "WIN"
        )


        losses = sum(
            1
            for t in trades
            if t.get("Status") == "LOSS"
        )


        break_even = sum(
            1
            for t in trades
            if t.get("Status")
            == "BREAK_EVEN"
        )


        closed = (
            wins
            + losses
            + break_even
        )


        # --------------------------------------------------------
        # WIN RATE
        # --------------------------------------------------------

        if closed > 0:

            win_rate = round(
                (
                    wins
                    / closed
                )
                * 100,
                2
            )

        else:

            win_rate = 0.0


        # --------------------------------------------------------
        # P&L
        # --------------------------------------------------------

        total_pnl = round(
            sum(
                float(
                    t.get(
                        "NetPnL",
                        t.get(
                            "PnL",
                            0
                        )
                    )
                )
                for t in trades
            ),
            2
        )


        total_return = round(
            (
                total_pnl
                / self.initial_capital
            )
            * 100,
            2
        )


        # --------------------------------------------------------
        # PROFITS / LOSSES
        # --------------------------------------------------------

        profits = [

            float(
                t.get(
                    "NetPnL",
                    t.get(
                        "PnL",
                        0
                    )
                )
            )

            for t in trades

            if float(
                t.get(
                    "NetPnL",
                    t.get(
                        "PnL",
                        0
                    )
                )
            ) > 0

        ]


        losses_list = [

            float(
                t.get(
                    "NetPnL",
                    t.get(
                        "PnL",
                        0
                    )
                )
            )

            for t in trades

            if float(
                t.get(
                    "NetPnL",
                    t.get(
                        "PnL",
                        0
                    )
                )
            ) < 0

        ]


        avg_profit = (

            round(
                sum(profits)
                / len(profits),
                2
            )

            if profits

            else 0.0

        )


        avg_loss = (

            round(
                sum(losses_list)
                / len(losses_list),
                2
            )

            if losses_list

            else 0.0

        )


        # --------------------------------------------------------
        # PROFIT FACTOR
        # --------------------------------------------------------

        gross_profit = sum(
            profits
        )


        gross_loss = abs(
            sum(
                losses_list
            )
        )


        if gross_loss > 0:

            profit_factor = round(
                gross_profit
                / gross_loss,
                2
            )

        elif gross_profit > 0:

            # No losing trades at all: the ratio is mathematically
            # infinite. Represent it as such (STEP 12) rather than
            # an arbitrary "999" - the dashboard renders float('inf')
            # as the infinity symbol only in this genuine case.
            profit_factor = float("inf")

        else:

            # No wins and no losses (e.g. only break-even trades,
            # or zero closed trades) - profit factor is undefined.
            profit_factor = None


        # --------------------------------------------------------
        # R MULTIPLES
        # --------------------------------------------------------

        win_r = [

            float(
                t.get(
                    "RMultiple",
                    0
                )
            )

            for t in trades

            if t.get(
                "Status"
            ) == "WIN"

        ]


        loss_r = [

            float(
                t.get(
                    "RMultiple",
                    0
                )
            )

            for t in trades

            if t.get(
                "Status"
            ) == "LOSS"

        ]


        avg_win_r = (

            round(
                sum(win_r)
                / len(win_r),
                2
            )

            if win_r

            else 0.0

        )


        avg_loss_r = (

            round(
                sum(loss_r)
                / len(loss_r),
                2
            )

            if loss_r

            else 0.0

        )


        r_values = [

            float(
                t.get(
                    "RMultiple",
                    0
                )
            )

            for t in trades

            if t.get(
                "Status"
            )
            in [
                "WIN",
                "LOSS"
            ]

        ]


        avg_r = (

            round(
                sum(r_values)
                / len(r_values),
                2
            )

            if r_values

            else 0.0

        )


        # --------------------------------------------------------
        # EXPECTANCY
        # --------------------------------------------------------

        if closed > 0:

            expectancy_dollar = round(

                (
                    wins
                    / closed
                    * avg_profit
                )

                +

                (
                    losses
                    / closed
                    * avg_loss
                ),

                2
            )


            expectancy_r = round(

                (
                    wins
                    / closed
                    * avg_win_r
                )

                +

                (
                    losses
                    / closed
                    * avg_loss_r
                ),

                2
            )

        else:

            expectancy_dollar = 0.0

            expectancy_r = 0.0


        # --------------------------------------------------------
        # HOLDING DAYS
        # --------------------------------------------------------

        holding_days = [

            float(
                t.get(
                    "HoldingDays",
                    0
                )
            )

            for t in trades

            if t.get(
                "HoldingDays",
                0
            ) is not None

        ]


        avg_holding_days = (

            round(
                sum(holding_days)
                / len(holding_days),
                1
            )

            if holding_days

            else 0.0

        )


        # --------------------------------------------------------
        # STREAKS
        # --------------------------------------------------------

        max_win_streak = 0

        max_loss_streak = 0

        current_win_streak = 0

        current_loss_streak = 0


        for trade in trades:

            status = trade.get(
                "Status"
            )


            if status == "WIN":

                current_win_streak += 1

                current_loss_streak = 0


                max_win_streak = max(

                    max_win_streak,

                    current_win_streak

                )


            elif status == "LOSS":

                current_loss_streak += 1

                current_win_streak = 0


                max_loss_streak = max(

                    max_loss_streak,

                    current_loss_streak

                )


            else:

                current_win_streak = 0

                current_loss_streak = 0


        # --------------------------------------------------------
        # MAX DRAWDOWN
        # --------------------------------------------------------

        max_drawdown = 0.0


        if equity_curve:

            equity_values = [

                float(
                    ec.get(
                        "Equity",
                        self.initial_capital
                    )
                )

                for ec in equity_curve

            ]


            peak = (
                self.initial_capital
            )


            for value in equity_values:

                peak = max(
                    peak,
                    value
                )


                drawdown = (
                    peak
                    - value
                )


                max_drawdown = max(
                    max_drawdown,
                    drawdown
                )


        else:

            equity = (
                self.initial_capital
            )

            peak = equity


            for trade in trades:

                equity += float(
                    trade.get(
                        "NetPnL",
                        trade.get(
                            "PnL",
                            0
                        )
                    )
                )


                peak = max(
                    peak,
                    equity
                )


                drawdown = (
                    peak
                    - equity
                )


                max_drawdown = max(
                    max_drawdown,
                    drawdown
                )


        max_drawdown = round(
            max_drawdown,
            2
        )


        # --------------------------------------------------------
        # TARGET HITS
        # --------------------------------------------------------

        target1_wins = sum(

            1

            for t in trades

            if t.get(
                "TargetHit"
            ) == "TARGET1"

        )


        target2_wins = sum(

            1

            for t in trades

            if t.get(
                "TargetHit"
            ) == "TARGET2"

        )


        target3_wins = sum(

            1

            for t in trades

            if t.get(
                "TargetHit"
            ) == "TARGET3"

        )


        # --------------------------------------------------------
        # DATA QUALITY
        # --------------------------------------------------------

        if closed == 0:

            data_quality = (
                "NO CLOSED TRADES"
            )

        elif (
            closed
            < self.min_trades_for_ranking
        ):

            data_quality = (
                "LOW SAMPLE"
            )

        else:

            data_quality = (
                "SUFFICIENT SAMPLE"
            )


        # --------------------------------------------------------
        # STRATEGY STATUS
        # --------------------------------------------------------

        if closed < 10:

            strategy_status = (
                "INSUFFICIENT DATA"
            )

        elif (
            profit_factor is not None
            and profit_factor > 1.5
            and expectancy_dollar > 0
            and win_rate > 50
        ):

            strategy_status = "STRONG"

        elif (
            profit_factor is not None
            and profit_factor > 1.0
            and expectancy_dollar > 0
        ):

            strategy_status = "PROFITABLE"

        else:

            strategy_status = "WEAK"


        # --------------------------------------------------------
        # FINAL EQUITY
        # --------------------------------------------------------

        final_equity = round(
            self.initial_capital
            + total_pnl,
            2
        )


        # --------------------------------------------------------
        # RETURN SUMMARY
        # --------------------------------------------------------

        return {

            "Total Trades":
                len(trades),

            "Wins":
                wins,

            "Losses":
                losses,

            "BreakEven":
                break_even,

            "Break Even":
                break_even,

            "Open":
                0,

            "Open Trades":
                0,

            "Closed Trades":
                closed,

            "Win Rate":
                win_rate,

            "Loss Rate":
                round(
                    (
                        losses
                        / closed
                    ) * 100,
                    2
                )
                if closed > 0
                else 0.0,

            "Total PnL":
                total_pnl,

            "Total Return %":
                total_return,

            "Average Profit":
                avg_profit,

            "Average Loss":
                avg_loss,

            "Avg Win R":
                avg_win_r,

            "Avg Loss R":
                avg_loss_r,

            "Average R":
                avg_r,

            "Profit Factor":
                profit_factor,

            "Expectancy":
                expectancy_dollar,

            "Expectancy R":
                expectancy_r,

            "Average Holding Days":
                avg_holding_days,

            "Max Drawdown":
                max_drawdown,

            "Best Trade":
                round(
                    max(
                        [
                            float(
                                t.get(
                                    "NetPnL",
                                    t.get(
                                        "PnL",
                                        0
                                    )
                                )
                            )
                            for t in trades
                        ]
                    ),
                    2
                ),

            "Worst Trade":
                round(
                    min(
                        [
                            float(
                                t.get(
                                    "NetPnL",
                                    t.get(
                                        "PnL",
                                        0
                                    )
                                )
                            )
                            for t in trades
                        ]
                    ),
                    2
                ),

            "Consecutive Wins":
                max_win_streak,

            "Consecutive Losses":
                max_loss_streak,

            "Target1 Wins":
                target1_wins,

            "Target2 Wins":
                target2_wins,

            "Target3 Wins":
                target3_wins,

            "Data Quality":
                data_quality,

            "Strategy Status":
                strategy_status,

            "Equity Curve":
                equity_curve,

            "Trades":
                trades,

            "FinalEquity":
                final_equity,

            "Debug":
                debug or {},

        }


    # ============================================================
    # EMPTY SUMMARY
    # ============================================================

    def _empty_summary(
        self
    ) -> Dict[str, Any]:
        """
        Return empty backtest summary.
        """

        return {

            "Total Trades":
                0,

            "Wins":
                0,

            "Losses":
                0,

            "BreakEven":
                0,

            "Break Even":
                0,

            "Open":
                0,

            "Open Trades":
                0,

            "Closed Trades":
                0,

            "Win Rate":
                0.0,

            "Loss Rate":
                0.0,

            "Total PnL":
                0.0,

            "Total Return %":
                0.0,

            "Average Profit":
                0.0,

            "Average Loss":
                0.0,

            "Avg Win R":
                0.0,

            "Avg Loss R":
                0.0,

            "Average R":
                0.0,

            "Profit Factor":
                None,

            "Expectancy":
                0.0,

            "Expectancy R":
                0.0,

            "Average Holding Days":
                0.0,

            "Max Drawdown":
                0.0,

            "Best Trade":
                0.0,

            "Worst Trade":
                0.0,

            "Consecutive Wins":
                0,

            "Consecutive Losses":
                0,

            "Target1 Wins":
                0,

            "Target2 Wins":
                0,

            "Target3 Wins":
                0,

            "Data Quality":
                "NO TRADES",

            "Strategy Status":
                "NO TRADES",

            "Equity Curve":
                [],

            "Trades":
                [],

            "FinalEquity":
                self.initial_capital,

            "Debug":
                {},

        }