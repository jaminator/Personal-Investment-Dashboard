"""
ui_holdings.py — Portfolio Holdings Management UI (Feature 1)
with per-holding dividend settings (DRIP toggle, yield, frequency).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import (
    compute_standardized_yield,
    detect_distribution_frequency,
    fetch_current_price,
    fetch_dividend_history,
)
from dividend_verifier import get_latest_distribution
from portfolio import ASSET_CLASSES, Holding, PortfolioConfig, default_portfolio


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def _pct(val: float) -> str:
    return f"{val:.2%}"


def render_holdings(config: PortfolioConfig) -> None:
    """Render the portfolio holdings management panel."""
    st.header("Portfolio Holdings")

    # --- Save / Load --------------------------------------------------------
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Download Portfolio JSON"):
            st.download_button(
                label="Download",
                data=config.to_json(),
                file_name="portfolio_config.json",
                mime="application/json",
                key="download_json",
            )
    with col_load:
        uploaded = st.file_uploader("Load Portfolio JSON", type=["json"], key="upload_json")
        if uploaded is not None:
            try:
                text = uploaded.read().decode("utf-8")
                loaded = PortfolioConfig.from_json(text)
                st.session_state["config"] = loaded
                st.success("Portfolio loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load portfolio: {e}")

    # --- Add holding --------------------------------------------------------
    st.subheader("Add Holding")
    with st.form("add_holding_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ticker = st.text_input("Ticker", value="", placeholder="e.g. AAPL").upper().strip()
        with c2:
            shares = st.number_input("Shares", min_value=0.0, value=10.0, step=1.0)
        with c3:
            cost_basis = st.number_input("Cost Basis ($)", min_value=0.0, value=100.0, step=1.0)
        with c4:
            asset_class = st.selectbox("Asset Class", ASSET_CLASSES)

        c5, c6, c7 = st.columns(3)
        with c5:
            is_manual = st.checkbox("Manual entry (no ticker)")
        with c6:
            manual_value = st.number_input("Manual Value ($)", min_value=0.0, value=0.0, step=100.0)
        with c7:
            manual_return = st.number_input("Expected Return (%)", min_value=-100.0, value=7.0, step=0.5) / 100.0

        submitted = st.form_submit_button("Add Holding")
        if submitted:
            h = Holding(
                ticker=ticker,
                shares=shares,
                cost_basis=cost_basis,
                asset_class=asset_class,
                is_manual=is_manual,
                manual_value=manual_value,
                manual_expected_return=manual_return,
            )
            if not is_manual and ticker:
                price = fetch_current_price(ticker)
                if price is not None:
                    h.current_price = price
                else:
                    st.warning(f"Could not fetch price for {ticker}. Set to $0.")
            config.holdings.append(h)
            st.rerun()

    # --- Fetch all prices ---------------------------------------------------
    for h in config.holdings:
        if not h.is_manual and h.ticker and h.current_price == 0:
            price = fetch_current_price(h.ticker)
            if price is not None:
                h.current_price = price

    # --- Holdings table -----------------------------------------------------
    if not config.holdings:
        st.info("No holdings yet. Add your first holding above.")
        return

    st.subheader("Current Holdings")
    total_val = config.total_value

    rows = []
    for h in config.holdings:
        # Compute dividend yield and frequency using verified layer
        div_yield_str = "—"
        freq_str = "—"
        verified_str = "—"
        if not h.is_manual and h.ticker:
            override = h.distributions_per_year_override
            y, dpy, freq_label = compute_standardized_yield(h.ticker, override_freq=override)
            if y > 0:
                div_yield_str = f"{y:.2%}"
            if freq_label != "Unknown":
                freq_str = freq_label
            # Check verification status from verified layer
            latest = get_latest_distribution(h.ticker)
            if latest:
                verified_str = "Y" if latest.amount_verified else "N"

        rows.append({
            "ID": h.id,
            "Ticker": h.ticker if not h.is_manual else f"[Manual] {h.ticker or 'N/A'}",
            "Shares": h.shares,
            "Cost Basis": _fmt(h.cost_basis),
            "Price": _fmt(h.current_price) if not h.is_manual else "N/A",
            "Market Value": _fmt(h.market_value),
            "Unrealized G/L ($)": _fmt(h.unrealized_gl),
            "Unrealized G/L (%)": _pct(h.unrealized_gl_pct),
            "Div Yield": div_yield_str,
            "Dist Freq": freq_str,
            "Verified": verified_str,
            "DRIP": "On" if h.drip_enabled else "Off",
            "Asset Class": h.asset_class,
            "% of Portfolio": _pct(h.market_value / total_val) if total_val > 0 else "0.00%",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    # --- Per-holding dividend settings --------------------------------------
    st.subheader("Dividend Settings")
    st.caption(
        "Configure DRIP (dividend reinvestment) and distribution "
        "frequency per holding.  Frequency is auto-detected from "
        "trailing 12-month dividend history; override if needed."
    )
    for h in config.holdings:
        if h.is_manual:
            continue
        with st.expander(f"{h.ticker} — Dividend Settings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                drip = st.toggle(
                    "Reinvest Distributions (DRIP)",
                    value=h.drip_enabled,
                    key=f"drip_{h.id}",
                    help="DRIP: dividends buy additional shares on payment date. "
                         "Off: dividends accrue to sleeve cash balance.",
                )
                h.drip_enabled = drip
            with c2:
                offset = st.number_input(
                    "Payment date offset (calendar days after ex-date)",
                    min_value=1,
                    max_value=60,
                    value=h.payment_date_offset,
                    step=1,
                    key=f"offset_{h.id}",
                    help="Used when FMP data is unavailable. "
                         "Monthly funds: ~15 days. Quarterly: ~20 days.",
                )
                h.payment_date_offset = offset

            c3, c4 = st.columns(2)
            with c3:
                freq_override = st.number_input(
                    "Override distributions/year (0 = auto-detect)",
                    min_value=0,
                    max_value=365,
                    value=h.distributions_per_year_override,
                    step=1,
                    key=f"freq_{h.id}",
                    help="Common values: 12 (monthly), 4 (quarterly), "
                         "2 (semi-annual), 1 (annual).",
                )
                h.distributions_per_year_override = freq_override
            with c4:
                # Show auto-detected frequency
                if h.ticker:
                    divs = fetch_dividend_history(h.ticker)
                    if not divs.empty:
                        label, dpy = detect_distribution_frequency(divs)
                        st.info(f"Detected: {label} ({dpy}x/year)")
                    else:
                        st.info("No dividend history found")

    # --- Remove holdings ----------------------------------------------------
    st.subheader("Remove Holdings")
    remove_options = {f"{h.ticker or 'Manual'} ({h.id})": h.id for h in config.holdings}
    to_remove = st.multiselect("Select holdings to remove", list(remove_options.keys()))
    if st.button("Remove Selected") and to_remove:
        ids_to_remove = {remove_options[k] for k in to_remove}
        config.holdings = [h for h in config.holdings if h.id not in ids_to_remove]
        # Also remove from sleeves
        for s in config.sleeves:
            s.holding_ids = [hid for hid in s.holding_ids if hid not in ids_to_remove]
        st.rerun()

    # --- Allocation charts --------------------------------------------------
    st.subheader("Portfolio Allocation")
    col1, col2 = st.columns(2)

    with col1:
        # By asset class
        ac_data = {}
        for h in config.holdings:
            ac_data[h.asset_class] = ac_data.get(h.asset_class, 0.0) + h.market_value
        if ac_data:
            fig = px.pie(
                names=list(ac_data.keys()),
                values=list(ac_data.values()),
                title="Allocation by Asset Class",
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By holding
        hold_data = {(h.ticker or "Manual"): h.market_value for h in config.holdings if h.market_value > 0}
        if hold_data:
            fig = px.pie(
                names=list(hold_data.keys()),
                values=list(hold_data.values()),
                title="Allocation by Holding",
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    # --- Summary ------------------------------------------------------------
    st.metric("Total Portfolio Value", _fmt(total_val))
    total_cost = sum(h.total_cost for h in config.holdings)
    total_gl = total_val - total_cost
    gl_pct = total_gl / total_cost if total_cost > 0 else 0.0
    st.metric(
        "Total Unrealized Gain/Loss",
        _fmt(total_gl),
        delta=_pct(gl_pct),
    )
