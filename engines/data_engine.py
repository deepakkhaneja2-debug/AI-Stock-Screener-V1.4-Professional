import time
import logging
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataEngine:
    """Handles data fetching and validation from Yahoo Finance."""

    def __init__(self):
        self.retry_count = 3
        self.retry_delay = 2
        self.auto_adjust = True
        self.min_data_rows = 100
        self.required_columns = ["Open", "High", "Low", "Close", "Volume"]

    def download_stock(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "6mo"
    ) -> pd.DataFrame:
        """Download a single stock's data with retries."""
        for attempt in range(self.retry_count):
            try:
                data = yf.download(
                    symbol,
                    interval=interval,
                    period=period,
                    progress=False,
                    auto_adjust=self.auto_adjust,
                    threads=False
                )
                if data is None or data.empty:
                    raise ValueError("No data received")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if not all(col in data.columns for col in self.required_columns):
                    raise ValueError("Missing required columns")

                data = self.clean_data(data)
                if data.empty:
                    raise ValueError("Data empty after cleaning")

                return data

            except Exception as e:
                logger.warning(f"{symbol} attempt {attempt+1}/{self.retry_count}: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

        logger.error(f"Failed to download {symbol}")
        return pd.DataFrame()

    def download_multiple(
        self,
        symbols: List[str],
        interval: str = "1d",
        period: str = "6mo"
    ) -> Dict[str, pd.DataFrame]:
        """Download multiple stocks sequentially."""
        result = {}
        for symbol in symbols:
            data = self.download_stock(symbol, interval, period)
            if self.validate_data(data):
                result[symbol] = data
            else:
                logger.warning(f"{symbol} rejected after validation")
        return result

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Check if data meets minimum requirements."""
        if data is None or data.empty:
            return False
        if len(data) < self.min_data_rows:
            return False
        if not all(col in data.columns for col in self.required_columns):
            return False
        return True

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        if data is None or data.empty:
            return pd.DataFrame()

        data = data.copy()
        data = data[~data.index.duplicated(keep="last")]

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=numeric_cols)
        data = data[
            (data["Close"] > 0) &
            (data["High"] > 0) &
            (data["Low"] > 0)
        ]
        return data

    def get_market_index(self) -> Dict[str, pd.DataFrame]:
        """Fetch market index data."""
        market_data = {}
        if USE_NIFTY_FILTER:
            nifty = self.download_stock("^NSEI")
            if self.validate_data(nifty):
                market_data["NIFTY"] = nifty
        if USE_BANKNIFTY_FILTER:
            banknifty = self.download_stock("^NSEBANK")
            if self.validate_data(banknifty):
                market_data["BANKNIFTY"] = banknifty
        return market_data

    def load_symbols(self) -> List[str]:
        """Return 39 NSE stocks for scanning."""
        nifty_39 = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
            "SBIN.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "HCLTECH.NS", "BAJFINANCE.NS",
            "NESTLEIND.NS", "TATAMOTORS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS",
            "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS", "IOC.NS",
            "BPCL.NS", "GAIL.NS", "ADANIPORTS.NS", "DIVISLAB.NS", "DRREDDY.NS",
            "CIPLA.NS", "BRITANNIA.NS", "HINDALCO.NS", "EICHERMOT.NS"
        ]

        if WATCHLIST_ONLY:
            return nifty_39[:5]

        if SCANNER_MODE == "CASH":
            return nifty_39[:15]
        elif SCANNER_MODE == "FNO":
            return nifty_39[15:30]
        else:
            return nifty_39

    def scan_ready_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch and validate data for all symbols in the scan list."""
        symbols = self.load_symbols()
        logger.info(f"Scanning {len(symbols)} symbols")
        return self.download_multiple(
            symbols=symbols,
            interval=PRIMARY_TIMEFRAME,
            period="6mo"
        )