"""
ui_analytics.py — Analytics & Statistics Dashboard UI (Feature 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    annualized_cagr,
    annualized_volatility,
    beta_alpha,
    calmar_ratio,
    compute_analytics,
    correlation_matrix,
    cvar,
    drawdown_series,
    max_drawdown,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    treynor_ratio,
    var_historical,
    var_parametric,
)
from data_fetcher import (
    fetch_returns,
    fetch_risk_free_rate,
)
from portfolio import PortfolioConfig


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def render_analytics(config: PortfolioConfig) -> None:
    """Render the comprehensive analytics dashboard."""
    st.header("Analytics & Statistics")

    # Check for backtest data or run from config
    backtest_result = st.session_state.get("backtest_result")
    sim_result = st.session_state.get("sim_result")

    source_options = []
    if backtest_result and not backtest_result.portfolio_values.empty:
        source_options.append("Backtest Results")
    if sim_result:
        source_options.append("Simulation (Median Path)")

    if not source_options:
        st.info("Run a backtest or simulation first to see analytics.")
        return

    source = st.selectbox("Analytics Source", source_options)

    if source == "Backtest Results":
        value_series = backtest_result.portfolio_values
    else:
        dates = sim_result["dates"]
        values = sim_result["p50"]
        value_series = pd.Series(values, index=pd.DatetimeIndex(dates), name="Portfolio")

    if value_series.empty or len(value_series) < 10:
        st.warning("Not enough data for analytics.")
        return

    returns = value_series.pct_change().dropna()
    rf = fetch_risk_free_rate()

    # Benchmark
    bench_ticker = config.benchmark_tickers[0] if config.benchmark_tickers else "SPY"
    bench_select = st.selectbox(
        "Benchmark for risk-adjusted metrics",
        config.benchmark_tickers + ["SPY", "AGG", "QQQ"],
    )
    bench_ticker = bench_select

    metrics = compute_analytics(value_series, bench_ticker, rf)
    if not metrics:
        st.warning("Could not compute analytics.")
        return

    # ======================================================================
    # Return Metrics
    # ======================================================================
    st.subheader("Return Metrics")
    ret_metrics = {k: v for k, v in metrics.items()
                   if k in ("Total Return", "CAGR", "Best Year", "Worst Year", "Best Month", "Worst Month")}
    _display_metric_cards(ret_metrics)

    # ======================================================================
    # Risk Metrics
    # ======================================================================
    st.subheader("Risk Metrics")
    risk_metrics = {k: v for k, v in metrics.items()
                    if k in ("Annualized Volatility", "Max Drawdown", "Max DD Period",
                             "VaR 95% (Historical)", "VaR 99% (Historical)",
                             "VaR 95% (Parametric)", "CVaR 95%")}
    _display_metric_cards(risk_metrics)

    # ======================================================================
    # Risk-Adjusted Metrics
    # ======================================================================
    st.subheader("Risk-Adjusted Metrics")
    ra_metrics = {k: v for k, v in metrics.items()
                  if k in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                           "Beta", "Alpha", "R-Squared", "Treynor Ratio")}
    _display_metric_cards(ra_metrics)

    # ======================================================================
    # Correlation heatmap
    # ======================================================================
    tickers = config.tickers()
    if len(tickers) >= 2:
        st.subheader("Correlation Matrix")
        start_str = str(value_series.index[0].date()) if hasattr(value_series.index[0], 'date') else str(value_series.index[0])
        end_str = str(value_series.index[-1].date()) if hasattr(value_series.index[-1], 'date') else str(value_series.index[-1])
        rets_df = fetch_returns(tickers, start=start_str, end=end_str)
        if not rets_df.empty and len(rets_df.columns) >= 2:
            corr = correlation_matrix(rets_df)
            fig_corr = px.imshow(
                corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Holdings Correlation Matrix",
                text_auto=".2f",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ======================================================================
    # Rolling metrics
    # ======================================================================
    st.subheader("Rolling Metrics")
    col1, col2 = st.columns(2)

    with col1:
        r_sharpe = rolling_sharpe(returns, window=252, risk_free=rf)
        if not r_sharpe.empty:
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=r_sharpe.index, y=r_sharpe.values,
                line=dict(color="#2196F3"),
                name="Rolling Sharpe",
            ))
            fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rs.update_layout(
                title="Rolling 12-Month Sharpe Ratio",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                hovermode="x unified",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    with col2:
        r_vol = rolling_volatility(returns, window=252)
        if not r_vol.empty:
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=r_vol.index, y=r_vol.values,
                line=dict(color="#FF9800"),
                name="Rolling Volatility",
            ))
            fig_rv.update_layout(
                title="Rolling 12-Month Volatility",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                yaxis_tickformat=".1%",
                hovermode="x unified",
            )
            st.plotly_chart(fig_rv, use_container_width=True)

    # ======================================================================
    # Sleeve-level analytics
    # ======================================================================
    if backtest_result and backtest_result.sleeve_values:
        st.subheader("Sleeve-Level Analytics")
        sleeve_rows = []
        for sname, svals in backtest_result.sleeve_values.items():
            if len(svals) < 10:
                continue
            s_ret = svals.pct_change().dropna()
            dd, _, _ = max_drawdown(svals)
            sleeve_rows.append({
                "Sleeve": sname,
                "Total Return": f"{total_return(svals):.2%}",
                "CAGR": f"{annualized_cagr(svals):.2%}",
                "Volatility": f"{annualized_volatility(s_ret):.2%}",
                "Max Drawdown": f"{dd:.2%}",
                "Sharpe": f"{sharpe_ratio(s_ret, rf):.3f}",
            })
        if sleeve_rows:
            st.dataframe(pd.DataFrame(sleeve_rows), use_container_width=True, hide_index=True)


def _display_metric_cards(metrics: dict) -> None:
    """Display metrics as a formatted table with color coding."""
    rows = []
    for key, val in metrics.items():
        if isinstance(val, float):
            # Color coding
            is_good = val > 0
            if key in ("Annualized Volatility", "Max Drawdown", "VaR 95% (Historical)",
                       "VaR 99% (Historical)", "VaR 95% (Parametric)", "CVaR 95%"):
                is_good = False  # These are always "bad" values

            if "Ratio" in key or key in ("Beta", "Alpha", "R-Squared"):
                formatted = f"{val:.4f}"
            elif any(kw in key for kw in ("Return", "CAGR", "Volatility", "Drawdown", "VaR", "CVaR")):
                formatted = f"{val:.2%}"
            else:
                formatted = f"{val:.4f}"

            color = "green" if is_good else "red"
            rows.append({"Metric": key, "Value": formatted, "_color": color})
        else:
            rows.append({"Metric": key, "Value": str(val), "_color": "gray"})

    if rows:
        df = pd.DataFrame(rows)
        # Use markdown for color coding
        for _, row in df.iterrows():
            color = row["_color"]
            st.markdown(
                f"**{row['Metric']}**: "
                f"<span style='color:{color}'>{row['Value']}</span>",
                unsafe_allow_html=True,
            )
