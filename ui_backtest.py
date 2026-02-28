"""
ui_backtest.py — Backtesting UI (Feature 6) with dividend event display.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import compute_analytics, drawdown_series
from backtester import BacktestResult, run_backtest, run_yield_sensitivity
from portfolio import PortfolioConfig
from rebalancer import DividendRecord


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def render_backtest(config: PortfolioConfig) -> None:
    """Render the backtesting panel."""
    st.header("Backtesting Engine")

    if not config.tickers():
        st.info("Add holdings with tickers to run a backtest.")
        return

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Backtest Start",
            value=dt.date(2019, 1, 1),
            key="bt_start",
        )
        config.backtest_start = str(start)
    with col2:
        end = st.date_input(
            "Backtest End",
            value=dt.date.today(),
            key="bt_end",
        )
        config.backtest_end = str(end)

    # Benchmark selection
    benchmarks = st.multiselect(
        "Benchmark(s)",
        ["SPY", "AGG", "QQQ", "IWM", "VTI", "BND", "GLD"],
        default=config.benchmark_tickers,
    )
    config.benchmark_tickers = benchmarks

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            result = run_backtest(config)
        if result.portfolio_values.empty:
            st.error("Backtest returned no data. Check your tickers and date range.")
            return
        st.session_state["backtest_result"] = result

    # Display results
    result: BacktestResult | None = st.session_state.get("backtest_result")
    if result is None or result.portfolio_values.empty:
        return

    # --- Equity curve -------------------------------------------------------
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.portfolio_values.index,
        y=result.portfolio_values.values,
        name="Portfolio",
        line=dict(color="#2196F3", width=2),
    ))
    for bname, bvals in result.benchmark_values.items():
        fig.add_trace(go.Scatter(
            x=bvals.index,
            y=bvals.values,
            name=bname,
            line=dict(width=1, dash="dash"),
        ))
    fig.update_layout(
        title="Portfolio vs. Benchmark(s)",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Sleeve breakdown ---------------------------------------------------
    if result.sleeve_values:
        st.subheader("Sleeve Value Over Time")
        fig_s = go.Figure()
        for sname, svals in result.sleeve_values.items():
            fig_s.add_trace(go.Scatter(
                x=svals.index, y=svals.values,
                name=sname, stackgroup="one",
            ))
        fig_s.update_layout(
            title="Sleeve Value Breakdown",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
        )
        st.plotly_chart(fig_s, width="stretch")

    # --- Drawdown chart -----------------------------------------------------
    st.subheader("Drawdown (Underwater) Chart")
    dd = drawdown_series(result.portfolio_values)
    if not dd.empty:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy",
            line=dict(color="#F44336", width=1),
            name="Drawdown",
        ))
        fig_dd.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, width="stretch")

    # --- Trade log ----------------------------------------------------------
    if result.trade_log:
        st.subheader("Trade Log")

        # View toggle: summary vs full audit trail
        view_mode = st.radio(
            "Trade log detail", ["Summary", "Full Audit Trail"],
            horizontal=True, key="trade_log_view",
        )

        trade_rows = []
        for tr in result.trade_log:
            if view_mode == "Summary":
                trade_rows.append({
                    "Date": str(tr.date),
                    "Sleeve": tr.sleeve_name,
                    "Event": getattr(tr, "event_type", ""),
                    "Ticker": tr.ticker,
                    "Action": tr.direction,
                    "Shares": f"{tr.shares:.4f}",
                    "Amount": _fmt(tr.dollar_amount),
                    "Tx Cost": _fmt(tr.transaction_cost),
                    "Cash After": _fmt(getattr(tr, "cash_after", 0.0)),
                    "Pair ID": getattr(tr, "paired_trade_id", "")[:8] or "—",
                    "Signal": _fmt(tr.signal_amount) if tr.signal_amount else "—",
                    "Note": tr.note or "—",
                })
            else:
                trade_rows.append({
                    "Date": str(tr.date),
                    "Sleeve": tr.sleeve_name,
                    "Event": getattr(tr, "event_type", ""),
                    "Ticker": tr.ticker,
                    "Action": tr.direction,
                    "Shares Before": f"{getattr(tr, 'shares_before', 0.0):.4f}",
                    "Shares After": f"{getattr(tr, 'shares_after', 0.0):.4f}",
                    "Share Delta": f"{getattr(tr, 'share_delta', 0.0):+.4f}",
                    "Exec Price": _fmt(getattr(tr, "execution_price", 0.0)),
                    "Gross Value": _fmt(getattr(tr, "gross_trade_value", 0.0)),
                    "Tx Cost": _fmt(tr.transaction_cost),
                    "Net Cash Impact": f"${getattr(tr, 'net_cash_impact', 0.0):+,.2f}",
                    "Cash Before": _fmt(getattr(tr, "cash_before", 0.0)),
                    "Cash After": _fmt(getattr(tr, "cash_after", 0.0)),
                    "Funding": getattr(tr, "funding_source", "") or "—",
                    "Pair ID": getattr(tr, "paired_trade_id", "")[:8] or "—",
                    "Signal": _fmt(tr.signal_amount) if tr.signal_amount else "—",
                    "Guard": "YES" if getattr(tr, "guard_active", False) else "—",
                    "Shortfall": _fmt(getattr(tr, "shortfall_amount", 0.0)) if getattr(tr, "shortfall_amount", 0.0) > 0 else "—",
                    "Note": tr.note or "—",
                })
        df_trades = pd.DataFrame(trade_rows)
        st.dataframe(df_trades, width="stretch", hide_index=True)

        # Mode B signal visualization
        mode_b_trades = [t for t in result.trade_log if t.signal_amount > 0]
        if mode_b_trades:
            st.subheader("Signal vs. Trade Amount")
            fig_sig = go.Figure()
            sig_dates = [t.date for t in mode_b_trades]
            sig_vals = [t.signal_amount for t in mode_b_trades]
            trade_vals = [t.dollar_amount for t in mode_b_trades]

            fig_sig.add_trace(go.Scatter(
                x=sig_dates, y=sig_vals,
                name="Signal $", line=dict(color="blue"),
            ))
            fig_sig.add_trace(go.Scatter(
                x=sig_dates, y=trade_vals,
                name="Trade $", line=dict(color="orange"),
            ))

            # Add markers on equity curve
            colors = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
            for direction, color in colors.items():
                dir_trades = [t for t in mode_b_trades if t.direction == direction]
                if dir_trades:
                    fig.add_trace(go.Scatter(
                        x=[t.date for t in dir_trades],
                        y=[
                            float(result.portfolio_values.loc[
                                result.portfolio_values.index[
                                    result.portfolio_values.index.searchsorted(
                                        pd.Timestamp(t.date)
                                    ).clip(0, len(result.portfolio_values) - 1)
                                ]
                            ]) if len(result.portfolio_values) > 0 else 0
                            for t in dir_trades
                        ],
                        mode="markers",
                        name=f"{direction} Signal",
                        marker=dict(color=color, size=8, symbol="triangle-up" if direction == "BUY" else "triangle-down"),
                    ))

            fig_sig.update_layout(
                title="Signal Amount vs. Executed Trade",
                xaxis_title="Date",
                yaxis_title="$",
                yaxis_tickformat="$,.0f",
                hovermode="x unified",
            )
            st.plotly_chart(fig_sig, width="stretch")

    # --- Dividend log -------------------------------------------------------
    if result.dividend_log:
        st.subheader("Dividend / Distribution Log")
        total_divs = sum(dr.gross_amount for dr in result.dividend_log)
        st.metric("Total Dividends Received", _fmt(total_divs))

        div_rows = []
        for dr in result.dividend_log:
            div_rows.append({
                "Ex-Div Date": str(dr.ex_dividend_date),
                "Payment Date": str(dr.payment_date),
                "Pay Date Src": dr.payment_date_source,
                "Decl. Date": str(dr.declaration_date) if dr.declaration_date else "—",
                "Record Date": str(dr.record_date) if dr.record_date else "—",
                "Sleeve": dr.sleeve_name,
                "Ticker": dr.ticker,
                "Div/Share": f"${dr.dividend_per_share:.4f}",
                "Shares Held": f"{dr.shares_held:.4f}",
                "Gross $": _fmt(dr.gross_amount),
                "Treatment": dr.treatment,
                "DRIP Shares": f"{dr.drip_shares:.4f}" if dr.drip_shares > 0 else "—",
                "DRIP Price": _fmt(dr.drip_price) if dr.drip_price > 0 else "—",
                "Cash Added": _fmt(dr.cash_added) if dr.cash_added > 0 else "—",
            })
        df_divs = pd.DataFrame(div_rows)
        st.dataframe(df_divs, width="stretch", hide_index=True)

    # --- Yield sensitivity (Mode B) ----------------------------------------
    mode_b_sleeves = [s for s in config.sleeves if s.mode.startswith("B")]
    if mode_b_sleeves:
        st.subheader("Yield Sensitivity Analysis")
        for sleeve in mode_b_sleeves:
            with st.expander(f"Sensitivity: {sleeve.name}"):
                if st.button(f"Run Sensitivity for {sleeve.name}", key=f"sens_{sleeve.id}"):
                    with st.spinner("Running sensitivity analysis..."):
                        sens_df = run_yield_sensitivity(config, sleeve)
                    if not sens_df.empty:
                        st.dataframe(sens_df.style.format({
                            "CAGR (%)": "{:.2f}%",
                            "Volatility (%)": "{:.2f}%",
                            "Max Drawdown (%)": "{:.2f}%",
                            "Sharpe Ratio": "{:.3f}",
                        }), width="stretch", hide_index=True)

    # --- Analytics ----------------------------------------------------------
    st.subheader("Backtest Analytics")
    bench_ticker = benchmarks[0] if benchmarks else "SPY"
    metrics = compute_analytics(result.portfolio_values, bench_ticker)
    if metrics:
        _display_metrics(metrics)


def _display_metrics(metrics: dict) -> None:
    """Display analytics metrics in a formatted table."""
    rows = []
    for key, val in metrics.items():
        if isinstance(val, float):
            if "Ratio" in key or key in ("Beta", "Alpha", "R-Squared"):
                formatted = f"{val:.4f}"
            elif "%" in key or key in ("Total Return", "CAGR", "Annualized Volatility", "Max Drawdown"):
                formatted = f"{val:.2%}"
            elif "VaR" in key or "CVaR" in key:
                formatted = f"{val:.4%}"
            else:
                formatted = f"{val:.4f}"
        else:
            formatted = str(val)
        rows.append({"Metric": key, "Value": formatted})
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
