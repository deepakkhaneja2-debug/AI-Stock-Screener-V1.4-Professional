# ============================================
# AI STOCK SCANNER V1.4
# CONFIGURATION FILE
# ============================================

# Scanner Mode
SCANNER_MODE = "BOTH"        # CASH / FNO / BOTH

# Trading Style
TRADING_STYLE = "SWING"      # SWING / POSITION

# Accuracy Mode
ACCURACY_MODE = "BALANCED"   # AGGRESSIVE / BALANCED / CONSERVATIVE

# Timeframes
PRIMARY_TIMEFRAME = "1d"
CONFIRMATION_TIMEFRAME = "4h"

# Results
TOP_BUY_RESULTS = 10
TOP_SELL_RESULTS = 10

# Risk Management
RISK_PER_TRADE = 1.0         # Percentage
DEFAULT_RR = 3.0             # Risk : Reward

# Backtest Cost Settings
SLIPPAGE = 0.001             # 0.1% Slippage
BROKERAGE_PER_TRADE = 20     # ₹20 per completed trade
BROKERAGE_TYPE = "fixed"     # fixed / percentage
ENABLE_TRANSACTION_COST = True

# Capital
STARTING_CAPITAL = 100000

# Alerts
ENABLE_SOUND_ALERT = True
ENABLE_POPUP_ALERT = True
ENABLE_WATCHLIST_ALERT = True

# Market Filter
USE_NIFTY_FILTER = True
USE_BANKNIFTY_FILTER = True

# Indicators
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Watchlist
WATCHLIST_ONLY = False

# Signal Thresholds
BUY_SCORE = 60
SELL_SCORE = 60

# Strategy Engine Thresholds
STRATEGY_BUY_THRESHOLD = 65
STRATEGY_SELL_THRESHOLD = 65

# Confidence Engine Weights
CONFIDENCE_WEIGHT_STRATEGY = 0.40
CONFIDENCE_WEIGHT_TREND = 0.25
CONFIDENCE_WEIGHT_PATTERN = 0.20
CONFIDENCE_WEIGHT_VOLUME = 0.10
CONFIDENCE_WEIGHT_ATR = 0.05

# Backtest Settings
BACKTEST_LOOKAHEAD = 30
MIN_TRADES_FOR_RANKING = 5

# Trading Settings
ENTRY_ATR_BUFFER = 0.0
# Phase 10 finding: independent sweep (0.0/0.1/0.2/0.3/0.5 ATR) showed a
# monotonic degradation as the buffer grows - a larger buffer means the
# entry chases price further above the signal close before confirming,
# which mechanically produces worse fills. 0.0 (enter on breakout above
# the signal close itself, no extra confirmation cushion) was best on
# every metric tested on synthetic data. Re-validate against real data.
STOP_ATR_MULTIPLIER = 1.75
# Phase 10 finding: independent sweep (1.0-2.0 ATR) showed stops were
# too tight at 1.5 - losses were partly noise-driven stop-outs, not bad
# entries. Widening to 1.75 ATR (not the tested max of 2.0, which gave
# back some of the expectancy gain in the combined test) balanced fewer
# noise stop-outs against risk-per-trade. Re-validate against real data.
TRAILING_STOP_ATR = 2.0
BREAK_EVEN_AT_TARGET1 = True
TARGET1_R = 1.5
TARGET2_R = 2.5
TARGET3_R = 4.0

# ============================================
# BACKTEST ENTRY WINDOW
# ============================================
# How many candles ahead of a signal we allow the
# theoretical breakout entry to actually trigger.
# Too small = many valid signals never get an entry
# (this was a major source of "0 trades" stocks).
BACKTEST_ENTRY_WINDOW = 15

# ============================================
# SCORE-BASED ENTRY MODEL (STEP 5)
# ============================================
# Instead of requiring every condition to be true at once
# (all-or-nothing AND gate), each candle is scored 0-100
# across these weighted components. A trade is only taken
# if the total score clears MIN_SIGNAL_SCORE.
# Weights must sum to 100.
SCORE_WEIGHT_TREND = 25
SCORE_WEIGHT_MOMENTUM = 20
SCORE_WEIGHT_VOLUME = 15
SCORE_WEIGHT_PATTERN = 15
SCORE_WEIGHT_MARKET = 15
SCORE_WEIGHT_VOLATILITY = 10

MIN_SIGNAL_SCORE = 60          # BUY score threshold used by the backtest
BACKTEST_BUY_THRESHOLD = MIN_SIGNAL_SCORE
BACKTEST_SELL_THRESHOLD = MIN_SIGNAL_SCORE

# ============================================
# ATR VOLATILITY FILTER (STEP 7)
# ============================================
# ATRPercent = ATR / Close * 100
# Reject setups where volatility is too low (no meaningful
# movement) or too high (unstable / unreliable stops).
MIN_ATR_PERCENT = 0.5
MAX_ATR_PERCENT = 6.0

# ============================================
# VOLUME FILTER
# ============================================
MIN_VOLUME_RATIO = 0.8   # Volume / 20-period average volume

# ============================================
# MARKET REGIME FILTER (STEP 6)
# ============================================
# NIFTY Close > EMA20 > EMA50            -> BULLISH
# NIFTY Close < EMA20 < EMA50            -> BEARISH
# otherwise                              -> NEUTRAL
USE_MARKET_REGIME_FILTER = True
MARKET_BULLISH_SCORE_BONUS = 10     # added to BUY score in a bullish market
MARKET_BEARISH_SCORE_PENALTY = 15   # subtracted from BUY score in a bearish market
MARKET_NEUTRAL_MIN_SCORE_BONUS = 5  # extra score required (via penalty) in neutral markets

# ============================================
# PARTIAL EXITS (STEP 9)
# ============================================
# If disabled, the full position exits at whichever target
# (or stop) is hit first, which is the current architecture's
# native behaviour and is always consistent/realistic.
ENABLE_PARTIAL_EXITS = False
PARTIAL_EXIT_TARGET1_PCT = 0.25
PARTIAL_EXIT_TARGET2_PCT = 0.35
PARTIAL_EXIT_TARGET3_PCT = 0.40

# ============================================
# DIAGNOSTICS (STEP 18 / 19)
# ============================================
BACKTEST_DEBUG = True