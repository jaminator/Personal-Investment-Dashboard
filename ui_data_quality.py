"""
ui_data_quality.py — Data Quality dashboard panel for dividend verification.

Displays per-holding verified dividend data, cross-validation status,
and any data quality warnings.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from dividend_verifier import (
    DividendEvent,
    get_verified_dividend_events,
    get_latest_distribution,
    compute_annualized_yield,
    generate_diagnostic_report,
)
from portfolio import PortfolioConfig


def _fmt(val: float) -> str:
    return f"${val:,.4f}"


def render_data_quality(config: PortfolioConfig) -> None:
    """Render the Data Quality dashboard panel."""
    st.header("Data Quality — Dividend Verification")

    tickers = config.tickers()
    if not tickers:
        st.info("Add holdings with tickers to see data quality checks.")
        return

    # --- Summary row at top --------------------------------------------------
    total_holdings = len(tickers)
    all_verified = 0
    pay_date_warnings = 0
    amount_warnings = 0

    holding_status: list[dict] = []
    for t in tickers:
        latest = get_latest_distribution(t)
        if latest is None:
            holding_status.append({"ticker": t, "has_data": False})
            continue
        has_amt_verified = latest.amount_verified
        has_pay_verified = latest.payment_date_verified
        has_warnings = len(latest.data_quality_warnings) > 0
        if has_amt_verified and has_pay_verified:
            all_verified += 1
        if not has_pay_verified and latest.payment_date_source != "ESTIMATED":
            pay_date_warnings += 1
        if not has_amt_verified:
            amount_warnings += 1
        holding_status.append({
            "ticker": t,
            "has_data": True,
            "amount_verified": has_amt_verified,
            "pay_verified": has_pay_verified,
            "warnings": latest.data_quality_warnings,
        })

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Holdings", total_holdings)
    with col2:
        st.metric("Fully Verified", all_verified)
    with col3:
        st.metric("Payment Date Warnings", pay_date_warnings)
    with col4:
        st.metric("Amount Warnings", amount_warnings)

    # --- Unresolved warnings -------------------------------------------------
    all_warnings = []
    for hs in holding_status:
        if hs.get("warnings"):
            for w in hs["warnings"]:
                all_warnings.append(f"**{hs['ticker']}**: {w}")

    if all_warnings:
        st.warning("**Unresolved Data Quality Warnings:**\n\n" + "\n\n".join(all_warnings))

    # --- Refresh button ------------------------------------------------------
    if st.button("Refresh Dividend Data", type="primary"):
        # Clear all Streamlit caches related to dividends
        st.cache_data.clear()
        st.rerun()

    # --- Per-holding dividend table ------------------------------------------
    st.subheader("Recent Dividend Events by Holding")

    for t in tickers:
        with st.expander(f"{t} — Last 4 Dividend Events", expanded=False):
            two_years_ago = dt.date.today() - dt.timedelta(days=730)
            events = get_verified_dividend_events(t, start_date=two_years_ago)
            recent = events[-4:] if len(events) > 4 else events

            if not recent:
                st.info(f"No dividend data available for {t}.")
                continue

            rows = []
            for ev in recent:
                rows.append({
                    "Ticker": ev.ticker,
                    "Ex-Dividend Date": str(ev.ex_dividend_date) if ev.ex_dividend_date else "—",
                    "Dist/Share ($)": f"${ev.distribution_per_share:.4f}",
                    "Payment Date": str(ev.payment_date) if ev.payment_date else "—",
                    "Pay Date Source": ev.payment_date_source,
                    "Amount Verified": "Y" if ev.amount_verified else "N",
                    "Pay Date Verified": "Y" if ev.payment_date_verified else "N",
                    "Amount Source": ev.amount_source,
                    "Warnings": "; ".join(ev.data_quality_warnings) if ev.data_quality_warnings else "—",
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show current verified yield
            vy = compute_annualized_yield(t)
            if vy is not None:
                st.caption(f"Verified annualized yield: {vy:.2%}")
            else:
                st.caption("No verified yield available")

    # --- Full diagnostic report ----------------------------------------------
    st.subheader("Full Diagnostic Report")
    with st.expander("Cross-Source Comparison (Last 8 Events per Ticker)", expanded=False):
        if st.button("Generate Full Report", key="gen_diag_report"):
            with st.spinner("Fetching and cross-validating all sources..."):
                diag_df = generate_diagnostic_report(tickers, num_events=8)
            if diag_df.empty:
                st.info("No dividend data available for diagnostic report.")
            else:
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
                st.session_state["diag_report"] = diag_df

        # Show cached report if available
        cached_report = st.session_state.get("diag_report")
        if cached_report is not None and not cached_report.empty:
            st.dataframe(cached_report, use_container_width=True, hide_index=True)
