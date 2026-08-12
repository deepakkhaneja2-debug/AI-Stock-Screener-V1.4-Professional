import logging
from typing import Optional, Union

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class AlertEngine:
    """Sends alerts via Streamlit and placeholder channels."""

    def __init__(self) -> None:
        pass

    def buy_alert(self, symbol: str, price: float) -> None:
        try:
            st.success(f"🟢 BUY : {symbol} @ ₹{price}")
            logger.info(f"BUY alert for {symbol} at ₹{price}")
        except Exception as e:
            logger.error(f"BUY alert error: {e}")

    def sell_alert(self, symbol: str, price: float) -> None:
        try:
            st.error(f"🔴 SELL : {symbol} @ ₹{price}")
            logger.info(f"SELL alert for {symbol} at ₹{price}")
        except Exception as e:
            logger.error(f"SELL alert error: {e}")

    def watch_alert(self, symbol: str) -> None:
        try:
            st.warning(f"🟡 WATCH : {symbol}")
            logger.info(f"WATCH alert for {symbol}")
        except Exception as e:
            logger.error(f"WATCH alert error: {e}")

    def notify(self, signal: Union[str, None], symbol: str, price: Optional[float] = None) -> None:
        if signal is None:
            signal = ""
        signal = str(signal).strip().upper()
        try:
            if signal == "BUY":
                if price is None:
                    logger.warning(f"BUY alert for {symbol} without price")
                    price = 0.0
                self.buy_alert(symbol, price)
            elif signal == "SELL":
                if price is None:
                    logger.warning(f"SELL alert for {symbol} without price")
                    price = 0.0
                self.sell_alert(symbol, price)
            else:
                self.watch_alert(symbol)
        except Exception as e:
            logger.error(f"Notification error: {e}")

    def telegram_alert(self, message: str) -> bool:
        logger.info(f"Telegram would send: {message}")
        return True

    def whatsapp_alert(self, message: str) -> bool:
        logger.info(f"WhatsApp would send: {message}")
        return True

    def email_alert(self, message: str) -> bool:
        logger.info(f"Email would send: {message}")
        return True

    def process(self, signal: Union[str, None], symbol: str, price: Optional[float] = None) -> bool:
        self.notify(signal, symbol, price)

        if signal is None:
            signal = ""
        signal_str = str(signal).strip().upper()
        message = f"{signal_str} | {symbol}" + (f" | ₹{price}" if price is not None else "")
        self.telegram_alert(message)
        self.whatsapp_alert(message)
        self.email_alert(message)
        logger.info(f"Processed alert: {message}")
        return True

    def send_alert(
        self,
        symbol: str,
        signal: Union[str, None],
        confidence: Optional[float] = None,
        entry: Optional[float] = None,
        stoploss: Optional[float] = None,
        target: Optional[float] = None
    ) -> bool:
        price = entry if entry is not None else None
        return self.process(signal, symbol, price)
