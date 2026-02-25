"""
ui_simulation.py — Monte Carlo Simulation Settings & Results UI (Feature 4).
"""

from __future__ import annotations

import streamlit as st

from portfolio import PortfolioConfig
from simulation import portfolio_return_params


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def render_simulation_settings(config: PortfolioConfig) -> None:
    """Render the simulation configuration panel (sidebar-friendly)."""
    st.header("Simulation Settings")

    # Forecast horizon
    horizon_opts = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20, "30Y": 30, "Custom": 0}
    horizon = st.selectbox(
        "Forecast Horizon",
        list(horizon_opts.keys()),
        index=3,
    )
    if horizon == "Custom":
        config.forecast_years = st.number_input("Custom horizon (years)", 1, 50, 10)
    else:
        config.forecast_years = horizon_opts[horizon]

    # Number of simulations
    config.num_simulations = st.slider(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=config.num_simulations,
        step=100,
    )

    # Simulation method
    config.simulation_method = st.selectbox(
        "Simulation Method",
        ["GBM", "Bootstrap"],
        index=0 if config.simulation_method == "GBM" else 1,
        help="GBM = Geometric Brownian Motion. Bootstrap = resample historical returns.",
    )

    # Return source
    config.return_source = st.selectbox(
        "Return Assumptions",
        ["Historical", "Manual"],
        index=0 if config.return_source == "Historical" else 1,
    )

    if config.return_source == "Historical":
        config.lookback_period = st.selectbox(
            "Lookback Period",
            ["1Y", "3Y", "5Y", "10Y", "Max"],
            index=2,
        )
        if config.lookback_period == "Max":
            config.lookback_period = "max"
    else:
        config.manual_return = st.number_input(
            "Expected Annual Return (%)",
            min_value=-50.0, max_value=100.0,
            value=config.manual_return * 100,
            step=0.5,
        ) / 100.0

        config.manual_volatility = st.number_input(
            "Annual Volatility (%)",
            min_value=0.0, max_value=200.0,
            value=config.manual_volatility * 100,
            step=0.5,
        ) / 100.0

    # Target portfolio value
    config.target_value = st.number_input(
        "Target Portfolio Value ($)",
        min_value=0.0,
        value=config.target_value,
        step=10000.0,
        format="%.0f",
    )

    # Show derived params
    if config.tickers():
        with st.expander("Derived Return Parameters"):
            mu, sigma = portfolio_return_params(config)
            st.write(f"**Expected Annual Return:** {mu:.2%}")
            st.write(f"**Annual Volatility:** {sigma:.2%}")
            st.write(f"**Geometric Return (approx):** {mu - 0.5 * sigma**2:.2%}")
