import traceback
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from scanner import StockScanner
from dashboard import format_profit_factor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Scanner V1.4",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLIPBOARD COPY BUTTON
# ============================================================

def copy_button(label: str, text: str, key: str):
    """
    Render a browser clipboard copy button.
    """

    safe_text = json.dumps(str(text))

    html = f"""
    <button
        onclick='navigator.clipboard.writeText({safe_text})'
        style="
            width:100%;
            padding:10px 14px;
            border:none;
            border-radius:8px;
            background:#262730;
            color:white;
            font-size:15px;
            font-weight:600;
            cursor:pointer;
        "
    >
        {label}
    </button>
    """

    components.html(
        html,
        height=50
    )


# ============================================================
# RESULT TEXT FORMATTER
# ============================================================

def dataframe_to_text(df: pd.DataFrame) -> str:
    """
    Convert dataframe into clean copyable text.
    """

    if df is None or df.empty:
        return "No results found."

    return df.to_string(
        index=False
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main Streamlit application."""

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🤖 AI Stock Scanner V1.4"
    )

    st.success(
        "✅ App Running Successfully"
    )

    st.caption(
        "AI-powered stock scanning, signal analysis and backtesting dashboard"
    )

    # ========================================================
    # CREATE SCANNER ONCE PER SESSION
    # ========================================================

    if "scanner" not in st.session_state:

        try:

            st.session_state.scanner = StockScanner()

        except Exception:

            st.error(
                "❌ StockScanner initialization failed"
            )

            st.code(
                traceback.format_exc()
            )

            st.stop()

    scanner = st.session_state.scanner

    # ========================================================
    # CONTROL PANEL
    # ========================================================

    st.subheader(
        "⚙️ Scanner Control"
    )

    col1, col2 = st.columns(2)

    with col1:

        run_scanner = st.button(
            "🚀 Run Scanner",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear_results = st.button(
            "🗑️ Clear Results",
            use_container_width=True
        )

    # ========================================================
    # CLEAR RESULTS
    # ========================================================

    if clear_results:

        st.session_state.pop(
            "scanner_results",
            None
        )

        st.session_state.pop(
            "backtest_results",
            None
        )

        try:

            st.session_state.scanner = StockScanner()

        except Exception:

            st.error(
                "❌ Scanner could not be reset"
            )

            st.code(
                traceback.format_exc()
            )

            st.stop()

        st.rerun()

    # ========================================================
    # RUN SCANNER
    # ========================================================

    if run_scanner:

        with st.spinner(
            "📡 Loading market data and running scanner..."
        ):

            try:

                results = scanner.run()

                # ------------------------------------------------
                # NORMALIZE RESULTS
                # ------------------------------------------------

                if results is None:

                    results = pd.DataFrame()

                elif not isinstance(
                    results,
                    pd.DataFrame
                ):

                    try:

                        results = pd.DataFrame(
                            results
                        )

                    except Exception:

                        results = pd.DataFrame()

                # ------------------------------------------------
                # STORE RESULTS
                # ------------------------------------------------

                st.session_state.scanner_results = (
                    results
                )

                st.session_state.backtest_results = (
                    getattr(
                        scanner,
                        "backtest_results",
                        {}
                    )
                )

                st.success(
                    "✅ Scanner Completed Successfully"
                )

            except Exception:

                st.error(
                    "❌ Scanner failed"
                )

                st.code(
                    traceback.format_exc()
                )

    # ========================================================
    # GET STORED RESULTS
    # ========================================================

    results = st.session_state.get(
        "scanner_results",
        pd.DataFrame()
    )

    backtest_results = st.session_state.get(
        "backtest_results",
        {}
    )

    # ========================================================
    # SCANNER RESULTS
    # ========================================================

    st.subheader(
        "📋 Scanner Results"
    )

    if (
        isinstance(results, pd.DataFrame)
        and not results.empty
    ):

        # ----------------------------------------------------
        # MAIN TABLE
        # ----------------------------------------------------

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PREPARE SIGNAL DATA
        # ----------------------------------------------------

        if "Signal" in results.columns:

            signal_series = (
                results["Signal"]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            buy_df = results[
                signal_series == "BUY"
            ].copy()

            sell_df = results[
                signal_series == "SELL"
            ].copy()

            watch_df = results[
                signal_series == "WATCH"
            ].copy()

        else:

            buy_df = pd.DataFrame()
            sell_df = pd.DataFrame()
            watch_df = pd.DataFrame()

        # ----------------------------------------------------
        # COPY TEXT
        # ----------------------------------------------------

        all_text = dataframe_to_text(
            results
        )

        buy_text = dataframe_to_text(
            buy_df
        )

        sell_text = dataframe_to_text(
            sell_df
        )

        watch_text = dataframe_to_text(
            watch_df
        )

        # ====================================================
        # COPY / EXPORT PANEL
        # ====================================================

        st.markdown(
            "### 📋 Copy / Export Results"
        )

        # ----------------------------------------------------
        # COPY BUTTONS
        # ----------------------------------------------------

        copy1, copy2, copy3, copy4 = st.columns(4)

        with copy1:

            copy_button(
                "📋 Copy ALL",
                all_text,
                "copy_all"
            )

        with copy2:

            copy_button(
                "📈 Copy BUY",
                buy_text,
                "copy_buy"
            )

        with copy3:

            copy_button(
                "📉 Copy SELL",
                sell_text,
                "copy_sell"
            )

        with copy4:

            copy_button(
                "👀 Copy WATCH",
                watch_text,
                "copy_watch"
            )

        # ----------------------------------------------------
        # DOWNLOAD BUTTONS
        # ----------------------------------------------------

        st.markdown(
            "#### ⬇️ Download"
        )

        download1, download2, download3, download4 = (
            st.columns(4)
        )

        with download1:

            st.download_button(
                "📄 ALL TXT",
                data=all_text,
                file_name="AI_Stock_Scanner_All.txt",
                mime="text/plain",
                use_container_width=True
            )

        with download2:

            st.download_button(
                "📈 BUY TXT",
                data=buy_text,
                file_name="AI_Stock_Scanner_BUY.txt",
                mime="text/plain",
                use_container_width=True
            )

        with download3:

            st.download_button(
                "📉 SELL TXT",
                data=sell_text,
                file_name="AI_Stock_Scanner_SELL.txt",
                mime="text/plain",
                use_container_width=True
            )

        with download4:

            st.download_button(
                "👀 WATCH TXT",
                data=watch_text,
                file_name="AI_Stock_Scanner_WATCH.txt",
                mime="text/plain",
                use_container_width=True
            )

        # ----------------------------------------------------
        # CSV DOWNLOAD
        # ----------------------------------------------------

        csv_data = results.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Complete Scanner Results CSV",
            data=csv_data,
            file_name="AI_Stock_Scanner_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ----------------------------------------------------
        # QUICK COUNTS
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Signal Count"
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Total",
            len(results)
        )

        q2.metric(
            "BUY",
            len(buy_df)
        )

        q3.metric(
            "SELL",
            len(sell_df)
        )

        q4.metric(
            "WATCH",
            len(watch_df)
        )

    else:

        st.info(
            "No scanner results yet. "
            "Click 'Run Scanner' to start."
        )

    # ========================================================
    # BACKTEST DASHBOARD
    # ========================================================

    st.subheader(
        "📊 Overall Backtest Dashboard"
    )

    if not backtest_results:

        st.info(
            "No backtest results available. "
            "Run the scanner first."
        )

    else:

        dashboard = getattr(
            scanner,
            "dashboard",
            None
        )

        if dashboard is None:

            st.warning(
                "⚠️ Dashboard module is not available."
            )

        else:

            try:

                overall_stats = (
                    dashboard.overall_stats(
                        backtest_results
                    )
                )

                if not isinstance(
                    overall_stats,
                    dict
                ):

                    overall_stats = {}

                total_trades = int(
                    overall_stats.get(
                        "Total Trades",
                        0
                    )
                )

                if total_trades > 0:

                    # --------------------------------------------
                    # ROW 1
                    # --------------------------------------------

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )

                    col1.metric(
                        "Total Trades",
                        total_trades
                    )

                    col2.metric(
                        "Wins",
                        int(
                            overall_stats.get(
                                "Wins",
                                0
                            )
                        )
                    )

                    col3.metric(
                        "Losses",
                        int(
                            overall_stats.get(
                                "Losses",
                                0
                            )
                        )
                    )

                    col4.metric(
                        "Win Rate",
                        f"{overall_stats.get('Win Rate', 0)}%"
                    )

                    # --------------------------------------------
                    # ROW 2
                    # --------------------------------------------

                    col5, col6, col7 = (
                        st.columns(3)
                    )

                    col5.metric(
                        "Total P&L",
                        round(
                            float(
                                overall_stats.get(
                                    "Total PnL",
                                    0
                                )
                            ),
                            2
                        )
                    )

                    col6.metric(
                        "Profit Factor",
                        format_profit_factor(
                            overall_stats.get(
                                "Profit Factor",
                                None
                            )
                        )
                    )

                    col7.metric(
                        "AI Score",
                        int(
                            overall_stats.get(
                                "AI Score",
                                0
                            )
                        )
                    )

                else:

                    st.info(
                        "Backtest completed but no "
                        "closed trades were generated."
                    )

            except Exception:

                st.warning(
                    "⚠️ Overall dashboard could not be generated."
                )

                st.code(
                    traceback.format_exc()
                )

    # ========================================================
    # STOCK RANKING
    # ========================================================

    st.subheader(
        "🏆 Stock Ranking"
    )

    dashboard = getattr(
        scanner,
        "dashboard",
        None
    )

    if dashboard is None:

        st.info(
            "Stock ranking dashboard is not available."
        )

    else:

        try:

            ranking_result = (
                dashboard.ranking_table(
                    backtest_results
                )
            )

            ranked_df = ranking_result.get("ranked") if isinstance(ranking_result, dict) else ranking_result
            low_sample_df = ranking_result.get("low_sample") if isinstance(ranking_result, dict) else None

            if (
                ranked_df is not None
                and not ranked_df.empty
            ):

                st.dataframe(
                    ranked_df,
                    use_container_width=True,
                    hide_index=True
                )

                highlights = dashboard.highlights(backtest_results)
                h1, h2, h3, h4, h5 = st.columns(5)
                h1.metric("Strongest Stock", highlights.get("Strongest Stock") or "-")
                h2.metric("Best AI Score", highlights.get("Best AI Score") or "-")
                h3.metric("Best PnL", highlights.get("Best PnL") or "-")
                h4.metric("Highest Win Rate", highlights.get("Highest Win Rate") or "-")
                h5.metric("Most Reliable", highlights.get("Most Reliable Stock") or "-")

            else:

                st.info(
                    "No stock ranking available. "
                    f"Stocks need at least {getattr(scanner.backtest_analyzer, 'min_trades_for_ranking', 5)} "
                    "closed trades to be ranked."
                )

            if low_sample_df is not None and not low_sample_df.empty:

                with st.expander(
                    "⚠️ Low Sample Stocks (below minimum trade count - not reliable for ranking)"
                ):
                    st.dataframe(
                        low_sample_df,
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception:

            st.warning(
                "⚠️ Stock ranking could not be generated."
            )

            st.code(
                traceback.format_exc()
            )

    # ========================================================
    # BACKTEST SUMMARY
    # ========================================================

    st.subheader(
        "📈 Backtest Summary"
    )

    if not backtest_results:

        st.info(
            "No individual backtest reports available."
        )

    else:

        for symbol, report in backtest_results.items():

            with st.expander(
                f"📊 {symbol}"
            ):

                # --------------------------------------------
                # VALIDATE REPORT
                # --------------------------------------------

                if not isinstance(
                    report,
                    dict
                ):

                    st.warning(
                        "Backtest report format incorrect."
                    )

                    continue

                # --------------------------------------------
                # SUMMARY
                # --------------------------------------------

                summary = {

                    "Total Trades": report.get(
                        "Total Trades",
                        0
                    ),

                    "Wins": report.get(
                        "Wins",
                        0
                    ),

                    "Losses": report.get(
                        "Losses",
                        0
                    ),

                    "BreakEven": report.get(
                        "BreakEven",
                        0
                    ),

                    "Win Rate": report.get(
                        "Win Rate",
                        0
                    ),

                    "Total PnL": report.get(
                        "Total PnL",
                        0
                    ),

                    "Total Return %": report.get(
                        "Total Return %",
                        0
                    ),

                    "Average Profit": report.get(
                        "Average Profit",
                        0
                    ),

                    "Average Loss": report.get(
                        "Average Loss",
                        0
                    ),

                    "Avg Win R": report.get(
                        "Avg Win R",
                        0
                    ),

                    "Avg Loss R": report.get(
                        "Avg Loss R",
                        0
                    ),

                    "Profit Factor": format_profit_factor(report.get(
                        "Profit Factor",
                        None
                    )),

                    "Expectancy": report.get(
                        "Expectancy",
                        0
                    ),

                    "Expectancy R": report.get(
                        "Expectancy R",
                        0
                    ),

                    "Average R": report.get(
                        "Average R",
                        0
                    ),

                    "Average Holding Days": report.get(
                        "Average Holding Days",
                        0
                    ),

                    "Max Drawdown": report.get(
                        "Max Drawdown",
                        0
                    ),

                    "Consecutive Wins": report.get(
                        "Consecutive Wins",
                        0
                    ),

                    "Consecutive Losses": report.get(
                        "Consecutive Losses",
                        0
                    ),

                    "Target1 Wins": report.get(
                        "Target1 Wins",
                        0
                    ),

                    "Target2 Wins": report.get(
                        "Target2 Wins",
                        0
                    ),

                    "Target3 Wins": report.get(
                        "Target3 Wins",
                        0
                    ),

                    "Data Quality": report.get(
                        "Data Quality",
                        "UNKNOWN"
                    ),

                    "Final Equity": report.get(
                        "FinalEquity",
                        0
                    )
                }

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                c1.metric(
                    "Trades",
                    int(
                        summary["Total Trades"]
                    )
                )

                c2.metric(
                    "Win Rate",
                    f"{summary['Win Rate']}%"
                )

                c3.metric(
                    "P&L",
                    round(
                        float(
                            summary["Total PnL"]
                        ),
                        2
                    )
                )

                c4.metric(
                    "AI Score",
                    int(
                        report.get(
                            "AI Score",
                            0
                        )
                    )
                )

                # --------------------------------------------
                # DETAILED SUMMARY TABLE
                # --------------------------------------------

                st.dataframe(
                    pd.DataFrame(
                        [summary]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------------
                # EQUITY CURVE
                # --------------------------------------------

                equity_curve = report.get(
                    "Equity Curve",
                    []
                )

                if (
                    isinstance(equity_curve, list)
                    and equity_curve
                ):

                    try:

                        equity_df = pd.DataFrame(
                            equity_curve
                        )

                        if (
                            "Date" in equity_df.columns
                            and "Equity" in equity_df.columns
                        ):

                            equity_df["Date"] = pd.to_datetime(
                                equity_df["Date"],
                                errors="coerce"
                            )

                            equity_df = (
                                equity_df
                                .dropna(
                                    subset=["Date"]
                                )
                                .set_index("Date")
                            )

                            if not equity_df.empty:

                                st.line_chart(
                                    equity_df[
                                        ["Equity"]
                                    ],
                                    use_container_width=True
                                )

                    except Exception:

                        st.caption(
                            "Equity curve could not be rendered "
                            "for this symbol."
                        )

                # --------------------------------------------
                # TRADES TABLE
                # --------------------------------------------

                trades = report.get(
                    "Trades",
                    []
                )

                if (
                    isinstance(trades, list)
                    and trades
                ):

                    st.markdown(
                        "#### 📋 Trade History"
                    )

                    try:

                        trades_df = pd.DataFrame(
                            trades
                        )

                        st.dataframe(
                            trades_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # ----------------------------------------
                        # COPY INDIVIDUAL STOCK TRADES
                        # ----------------------------------------

                        trades_text = (
                            trades_df.to_string(
                                index=False
                            )
                        )

                        copy_button(
                            f"📋 Copy {symbol} Trade History",
                            trades_text,
                            f"copy_trades_{symbol}"
                        )

                        st.download_button(
                            f"⬇️ Download {symbol} Trades CSV",
                            data=trades_df.to_csv(
                                index=False
                            ),
                            file_name=(
                                f"{symbol}_backtest_trades.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception:

                        st.warning(
                            "Trade history could not be displayed."
                        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
           