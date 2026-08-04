#!/usr/bin/env python3
"""
Standalone scanner script for AI Stock Scanner V1.4.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from engines.data_engine import DataEngine


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    u = d.clip(lower=0)
    l = -d.clip(upper=0)
    rs = u.rolling(p).mean() / l.rolling(p).mean()
    return 100 - 100 / (1 + rs)


def main() -> None:
    print("🤖 AI Stock Scanner V1.4")
    print("=" * 40)
    print("Loading symbols...")
    
    engine = DataEngine()
    stocks = engine.load_symbols()
    print(f"📊 Scanning {len(stocks)} stocks...\n")
    
    rows = []
    for st in stocks:
        try:
            print(f"  🔄 Processing {st}...", end=" ")
            df = yf.download(st, period="6mo", progress=False, auto_adjust=True)
            if df.empty:
                print("❌ No data")
                continue
            
            c = df["Close"].squeeze()
            e20 = c.ewm(span=20).mean().iloc[-1]
            e50 = c.ewm(span=50).mean().iloc[-1]
            rv = rsi(c).iloc[-1]
            
            if c.iloc[-1] > e20 > e50:
                sig = "🟢 BUY"
            elif c.iloc[-1] < e20 < e50:
                sig = "🔴 SELL"
            else:
                sig = "🟡 WATCH"
            
            rows.append([st, round(c.iloc[-1], 2), round(e20, 2), round(e50, 2), round(rv, 2), sig])
            print(f"✅ {sig} @ ₹{round(c.iloc[-1], 2)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    out = pd.DataFrame(rows, columns=["Stock", "Price", "EMA20", "EMA50", "RSI", "Signal"])
    print(f"\n✅ Scan complete. Found {len(out)} stocks with data.")
    
    out.to_excel("NSE_Scanner.xlsx", index=False)
    print("📁 Results saved to NSE_Scanner.xlsx")
    print("\n📊 Summary:")
    print(out.groupby("Signal").size())


if __name__ == "__main__":
    main()