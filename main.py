"""
main.py — Investment Portfolio Dashboard entry point.

Run with: streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

from data_fetcher import validate_fmp_key
from portfolio import PortfolioConfig, default_portfolio
from ui_analytics import render_analytics
from ui_backtest import render_backtest
from ui_contributions import render_contributions
from ui_future_value import render_future_value
from ui_holdings import render_holdings
from ui_simulation import render_simulation_settings
from ui_data_quality import render_data_quality
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
            "Data Quality",
        ],
        index=0,
    )

    st.markdown("---")
    render_simulation_settings(config)

    # --- FMP API Key Configuration -----------------------------------------
    st.markdown("---")
    st.subheader("API Keys")

    # Read existing key from secrets or session state
    existing_key = ""
    try:
        existing_key = st.secrets.get("FMP_API_KEY", "")
    except Exception:
        pass
    if not existing_key:
        existing_key = st.session_state.get("fmp_api_key", "")

    fmp_key = st.text_input(
        "FMP API Key",
        value=existing_key,
        type="password",
        help="Financial Modeling Prep free-tier key (250 req/day). "
             "Used for accurate dividend payment dates.",
        key="fmp_key_input",
    )
    if fmp_key != st.session_state.get("fmp_api_key", ""):
        st.session_state["fmp_api_key"] = fmp_key

    # Show FMP connection status
    if fmp_key:
        status = validate_fmp_key(fmp_key)
        if status == "valid":
            st.markdown(":white_check_mark: FMP connected")
        elif status == "invalid":
            st.markdown(":red_circle: FMP key invalid or rate-limited")
        else:
            st.markdown(":red_circle: FMP connection error")
    else:
        st.markdown(":warning: No FMP key — using estimated payment dates")

    st.markdown("---")
    st.caption("Investment Portfolio Dashboard v3.0 — Verified Dividends")

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
elif page == "Data Quality":
    render_data_quality(config)
