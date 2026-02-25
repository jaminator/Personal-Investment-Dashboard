"""
main.py — Investment Portfolio Dashboard entry point.

Run with: streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

from portfolio import PortfolioConfig, default_portfolio
from ui_analytics import render_analytics
from ui_backtest import render_backtest
from ui_contributions import render_contributions
from ui_future_value import render_future_value
from ui_holdings import render_holdings
from ui_simulation import render_simulation_settings
from ui_sleeves import render_sleeves


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investment Portfolio Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "config" not in st.session_state:
    st.session_state["config"] = default_portfolio()

config: PortfolioConfig = st.session_state["config"]

# ---------------------------------------------------------------------------
# Sidebar — navigation + simulation settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Portfolio Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Holdings",
            "Cash Flows",
            "Sleeves",
            "Future Value",
            "Backtest",
            "Analytics",
        ],
        index=0,
    )

    st.markdown("---")
    render_simulation_settings(config)

    st.markdown("---")
    st.caption("Investment Portfolio Dashboard v1.0")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if page == "Holdings":
    render_holdings(config)
elif page == "Cash Flows":
    render_contributions(config)
elif page == "Sleeves":
    render_sleeves(config)
elif page == "Future Value":
    render_future_value(config)
elif page == "Backtest":
    render_backtest(config)
elif page == "Analytics":
    render_analytics(config)
