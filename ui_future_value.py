"""
ui_future_value.py — Future Value Visualization UI (Feature 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio import PortfolioConfig
from simulation import run_simulation


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def render_future_value(config: PortfolioConfig) -> None:
    """Render the future value projection panel."""
    st.header("Future Value Projection")

    if config.total_value <= 0 and not config.contributions:
        st.info("Add holdings or contributions to see future value projections.")
        return

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {config.num_simulations:,} simulations..."):
            result = run_simulation(config)
        st.session_state["sim_result"] = result

    result = st.session_state.get("sim_result")
    if result is None:
        st.info("Click 'Run Monte Carlo Simulation' to generate projections.")
        return

    # ======================================================================
    # Summary metrics
    # ======================================================================
    st.subheader("Projection Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Starting Value", _fmt(result["initial_value"]))
    with col2:
        st.metric("Median Ending Value", _fmt(result["median_ending"]))
    with col3:
        st.metric(f"P(Reach {_fmt(config.target_value)})", f"{result['prob_target']:.1%}")
    with col4:
        st.metric("P(Ruin)", f"{result['prob_ruin']:.1%}")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Total Contributions", _fmt(result["total_contributions"]))
    with col6:
        st.metric("Total Withdrawals", _fmt(result["total_withdrawals"]))
    with col7:
        st.metric("Est. Investment Returns", _fmt(result["estimated_returns"]))

    st.caption(
        f"Assumptions: {result['mu']:.2%} annual return, "
        f"{result['sigma']:.2%} volatility, "
        f"{config.num_simulations:,} simulations ({config.simulation_method})"
    )

    # ======================================================================
    # Waterfall chart
    # ======================================================================
    st.subheader("Value Waterfall")
    waterfall_data = {
        "Starting Value": result["initial_value"],
        "Contributions": result["total_contributions"],
        "Withdrawals": -result["total_withdrawals"],
        "Investment Returns": result["estimated_returns"],
        "Ending Value": result["median_ending"],
    }

    fig_wf = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=list(waterfall_data.keys()),
        y=list(waterfall_data.values()),
        textposition="outside",
        text=[_fmt(v) for v in waterfall_data.values()],
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
        increasing=dict(marker_color="#4CAF50"),
        decreasing=dict(marker_color="#F44336"),
        totals=dict(marker_color="#2196F3"),
    ))
    fig_wf.update_layout(
        title="Portfolio Value Waterfall",
        yaxis_title="Value ($)",
        yaxis_tickformat="$,.0f",
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Waterfall summary table
    wf_table = pd.DataFrame([
        {"Component": k, "Amount": _fmt(v)}
        for k, v in waterfall_data.items()
    ])
    st.dataframe(wf_table, use_container_width=True, hide_index=True)

    # ======================================================================
    # Portfolio Value Over Time Chart
    # ======================================================================
    st.subheader("Portfolio Value Over Time")

    show_paths = st.checkbox("Show individual simulation paths", value=False)

    dates = result["dates"]
    fig_proj = go.Figure()

    # Confidence bands
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["p90"],
        line=dict(width=0), showlegend=False,
        name="90th percentile",
        hovertemplate="%{y:$,.0f}",
    ))
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["p10"],
        fill="tonexty",
        fillcolor="rgba(33, 150, 243, 0.1)",
        line=dict(width=0),
        name="10th-90th percentile",
        hovertemplate="%{y:$,.0f}",
    ))
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["p75"],
        line=dict(width=0), showlegend=False,
        name="75th percentile",
        hovertemplate="%{y:$,.0f}",
    ))
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["p25"],
        fill="tonexty",
        fillcolor="rgba(33, 150, 243, 0.2)",
        line=dict(width=0),
        name="25th-75th percentile",
        hovertemplate="%{y:$,.0f}",
    ))

    # Median line
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["p50"],
        line=dict(color="#2196F3", width=2),
        name="Median (50th)",
        hovertemplate="%{y:$,.0f}",
    ))

    # Mean line
    fig_proj.add_trace(go.Scatter(
        x=dates, y=result["mean"],
        line=dict(color="#FF9800", width=1, dash="dash"),
        name="Mean",
        hovertemplate="%{y:$,.0f}",
    ))

    # Individual paths
    if show_paths:
        n_paths = min(75, result["paths"].shape[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(result["paths"].shape[0], n_paths, replace=False)
        for i in idx:
            fig_proj.add_trace(go.Scatter(
                x=dates, y=result["paths"][i],
                line=dict(width=0.3, color="rgba(150,150,150,0.3)"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Withdrawal markers
    w_sched = result.get("withdrawal_schedule")
    if w_sched is not None and not w_sched.empty:
        for _, row in w_sched.iterrows():
            wd = row["date"]
            label = row.get("label", "")
            # Find approximate portfolio value at that date
            idx = min(
                range(len(dates)),
                key=lambda i: abs((dates[i] - wd).days) if hasattr(dates[i], 'days') else abs(
                    (pd.Timestamp(dates[i]) - pd.Timestamp(wd)).days
                ),
            )
            y_val = float(result["p50"][idx])
            fig_proj.add_trace(go.Scatter(
                x=[wd], y=[y_val],
                mode="markers+text",
                marker=dict(color="red", size=10, symbol="x"),
                text=[label or "Withdrawal"],
                textposition="top center",
                showlegend=False,
            ))

    # Target line
    fig_proj.add_hline(
        y=config.target_value,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Target: {_fmt(config.target_value)}",
        annotation_position="top right",
    )

    fig_proj.update_layout(
        title="Monte Carlo Portfolio Projection",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ======================================================================
    # Terminal value distribution
    # ======================================================================
    st.subheader("Terminal Value Distribution")
    final_values = result["paths"][:, -1]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_values,
        nbinsx=50,
        marker_color="#2196F3",
        opacity=0.7,
        name="Terminal Values",
    ))
    fig_hist.add_vline(
        x=float(np.median(final_values)),
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median: {_fmt(float(np.median(final_values)))}",
    )
    fig_hist.add_vline(
        x=config.target_value,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Target: {_fmt(config.target_value)}",
    )
    fig_hist.update_layout(
        title="Distribution of Ending Portfolio Values",
        xaxis_title="Portfolio Value ($)",
        xaxis_tickformat="$,.0f",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ======================================================================
    # Percentile table
    # ======================================================================
    st.subheader("Percentile Outcomes")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_vals = np.percentile(final_values, percentiles)
    pct_table = pd.DataFrame({
        "Percentile": [f"{p}th" for p in percentiles],
        "Ending Value": [_fmt(v) for v in pct_vals],
    })
    st.dataframe(pct_table, use_container_width=True, hide_index=True)

    # ======================================================================
    # Allocation over time (simplified)
    # ======================================================================
    if len(config.holdings) > 1 and any(not h.is_manual for h in config.holdings):
        st.subheader("Allocation Drift Over Time (No Rebalancing)")
        _render_allocation_drift(config, result)


def _render_allocation_drift(config: PortfolioConfig, sim_result: dict) -> None:
    """Show how allocation drifts over the projection period assuming
    each holding grows at the portfolio rate (simplified)."""
    dates = sim_result["dates"]
    n_dates = len(dates)
    p50 = sim_result["p50"]

    # Group holdings by asset class
    ac_values = {}
    total_val = config.total_value
    if total_val <= 0:
        return

    for h in config.holdings:
        ac = h.asset_class
        weight = h.market_value / total_val if total_val > 0 else 0
        if ac not in ac_values:
            ac_values[ac] = np.zeros(n_dates)
        ac_values[ac] += weight * p50

    fig = go.Figure()
    for ac, vals in ac_values.items():
        fig.add_trace(go.Scatter(
            x=dates, y=vals,
            name=ac,
            stackgroup="one",
            hovertemplate="%{y:$,.0f}",
        ))

    fig.update_layout(
        title="Projected Allocation Over Time by Asset Class",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
