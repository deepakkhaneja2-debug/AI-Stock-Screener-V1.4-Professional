import math


class BrokerageEngine:

    def __init__(self):

        # Zerodha-like default charges

        self.brokerage_percent = 0.0003
        self.max_brokerage = 20

        self.stt_percent = 0.001

        self.exchange_percent = 0.0000345

        self.sebi_percent = 0.000001

        self.gst_percent = 0.18

        self.stamp_percent = 0.00015

        self.default_slippage = 0.0005

    # ====================================
    # BROKERAGE
    # ====================================

    def brokerage(self, turnover):

        charge = turnover * self.brokerage_percent

        return min(charge, self.max_brokerage)

    # ====================================
    # SLIPPAGE
    # ====================================

    def slippage_price(self, price):

        return price * self.default_slippage

    # ====================================
    # TOTAL CHARGES
    # ====================================

    def total_charges(

        self,

        buy_price,

        sell_price,

        qty

    ):

        buy_turnover = buy_price * qty

        sell_turnover = sell_price * qty

        turnover = buy_turnover + sell_turnover

        brokerage = (

            self.brokerage(buy_turnover)

            +

            self.brokerage(sell_turnover)

        )

        stt = sell_turnover * self.stt_percent

        exchange = turnover * self.exchange_percent

        sebi = turnover * self.sebi_percent

        stamp = buy_turnover * self.stamp_percent

        gst = (

            brokerage +

            exchange

        ) * self.gst_percent

        total = (

            brokerage +

            stt +

            exchange +

            sebi +

            stamp +

            gst

        )

        return {

            "Brokerage": round(brokerage,2),

            "STT": round(stt,2),

            "Exchange": round(exchange,2),

            "GST": round(gst,2),

            "SEBI": round(sebi,2),

            "Stamp": round(stamp,2),

            "TotalCharges": round(total,2)

        }

    # ====================================
    # NET PNL
    # ====================================

    def net_pnl(

        self,

        buy_price,

        sell_price,

        qty

    ):

        gross = (

            sell_price -

            buy_price

        ) * qty

        charges = self.total_charges(

            buy_price,

            sell_price,

            qty

        )

        net = gross - charges["TotalCharges"]

        return {

            "GrossPnL": round(gross,2),

            "NetPnL": round(net,2),

            "Charges": charges

        }