"""
ui_data_quality.py — Data Quality dashboard panel for dividend verification.

Displays per-holding verified dividend data, FMP vs yfinance comparison,
and FMP connection status.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from data_fetcher import test_fmp_connection
from dividend_verifier import (
    DividendEvent,
    get_verified_dividend_events,
    get_latest_distribution,
    compute_annualized_yield,
    generate_diagnostic_report,
    _estimate_payment_date,
    _amounts_agree,
)
from portfolio import PortfolioConfig


def _fmt(val: float) -> str:
    return f"${val:,.4f}"


def render_data_quality(config: PortfolioConfig) -> None:
    """Render the Data Quality dashboard panel."""
    st.header("Data Quality \u2014 Dividend Verification")

    tickers = config.tickers()
    if not tickers:
        st.info("Add holdings with tickers to see data quality checks.")
        return

    # =================================================================
    # Section C: FMP Connection Status (at top for visibility)
    # =================================================================
    st.subheader("FMP Connection Status")

    fmp_status = test_fmp_connection()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if fmp_status["connected"]:
            st.metric("FMP API Status", "\u2705 Connected")
        else:
            st.metric("FMP API Status", "\u274c Error")
    with c2:
        remaining = fmp_status.get("rate_limit_remaining", "?")
        st.metric("Requests Remaining", f"{remaining} / 250")
    with c3:
        used = 250 - int(remaining) if isinstance(remaining, int) else "?"
        st.metric("Requests Used Today", str(used))
    with c4:
        if fmp_status.get("error_message"):
            st.metric("Error", fmp_status["error_message"][:40])
        else:
            st.metric("Last Status", "OK")

    if st.button("Refresh FMP Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # =================================================================
    # Section B: Data Quality Summary
    # =================================================================
    st.subheader("Data Quality Summary")

    total_holdings = len(tickers)
    fmp_pay_count = 0
    est_pay_count = 0
    amount_discrepancy_count = 0

    for t in tickers:
        latest = get_latest_distribution(t)
        if latest is None:
            est_pay_count += 1
            continue
        if latest.payment_date_source == "FMP":
            fmp_pay_count += 1
        else:
            est_pay_count += 1
        if not latest.amount_verified and latest.fmp_amount is not None and latest.yfinance_amount is not None:
            if not _amounts_agree(latest.yfinance_amount, latest.fmp_amount):
                amount_discrepancy_count += 1

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Holdings", total_holdings)
    with mc2:
        pct = f"{fmp_pay_count / total_holdings * 100:.0f}%" if total_holdings else "0%"
        st.metric("FMP Payment Dates", f"{fmp_pay_count} ({pct})")
    with mc3:
        pct = f"{est_pay_count / total_holdings * 100:.0f}%" if total_holdings else "0%"
        st.metric("Estimated Payment Dates", f"{est_pay_count} ({pct})")
    with mc4:
        st.metric("Amount Discrepancies", amount_discrepancy_count)

    # =================================================================
    # Section A: Per-Holding Dividend Summary
    # =================================================================
    st.subheader("Recent Dividend Events by Holding")

    for t in tickers:
        with st.expander(f"{t} \u2014 Last 4 Dividend Events", expanded=False):
            two_years_ago = dt.date.today() - dt.timedelta(days=730)
            events = get_verified_dividend_events(t, start_date=two_years_ago)
            recent = events[-4:] if len(events) > 4 else events

            if not recent:
                st.info(f"No dividend data available for {t}.")
                continue

            rows = []
            for ev in recent:
                amt_agree = "\u2014"
                if ev.fmp_amount is not None and ev.yfinance_amount is not None:
                    amt_agree = "\u2705" if _amounts_agree(ev.yfinance_amount, ev.fmp_amount) else "\u26a0\ufe0f"

                pay_src_indicator = "\u2705" if ev.payment_date_source == "FMP" else "\u26a0\ufe0f"

                est_pay = _estimate_payment_date(ev.ex_dividend_date, ev.frequency) if ev.ex_dividend_date else None

                rows.append({
                    "Ticker": ev.ticker,
                    "Ex-Dividend Date": str(ev.ex_dividend_date) if ev.ex_dividend_date else "\u2014",
                    "yfinance Amount ($)": f"${ev.yfinance_amount:.4f}" if ev.yfinance_amount is not None else "\u2014",
                    "FMP Amount ($)": f"${ev.fmp_amount:.4f}" if ev.fmp_amount is not None else "\u2014",
                    "Amount Used ($)": f"${ev.distribution_per_share:.4f}",
                    "Amount Source": ev.amount_source,
                    "Amount Agreement": amt_agree,
                    "FMP Payment Date": str(ev.fmp_payment_date) if ev.fmp_payment_date else "\u2014",
                    "Est. Payment Date": str(est_pay) if est_pay else "\u2014",
                    "Payment Date Used": str(ev.payment_date) if ev.payment_date else "\u2014",
                    "Pay Date Source": f"{pay_src_indicator} {ev.payment_date_source}",
                    "FMP Record Date": str(ev.fmp_record_date) if ev.fmp_record_date else "\u2014",
                    "FMP Declaration Date": str(ev.fmp_declaration_date) if ev.fmp_declaration_date else "\u2014",
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

            # Show frequency and yield
            if recent:
                freq = recent[0].frequency
                freq_src = recent[0].frequency_source
                freq_label = {12: "Monthly", 4: "Quarterly", 2: "Semi-Annual", 1: "Annual"}.get(freq, "Unknown")
                st.caption(f"Frequency: {freq_label} ({freq}x/year) — source: {freq_src}")

            vy = compute_annualized_yield(t)
            if vy is not None:
                st.caption(f"Verified annualized yield: {vy:.2%}")
            else:
                st.caption("No verified yield available")

    # =================================================================
    # Full Diagnostic Report
    # =================================================================
    st.subheader("Full Diagnostic Report")
    with st.expander("FMP vs yfinance Comparison (Last 8 Events per Ticker)", expanded=False):
        if st.button("Generate Full Report", key="gen_diag_report"):
            with st.spinner("Fetching and cross-validating FMP vs yfinance..."):
                diag_df = generate_diagnostic_report(tickers, num_events=8)
            if diag_df.empty:
                st.info("No dividend data available for diagnostic report.")
            else:
                st.dataframe(diag_df, width="stretch", hide_index=True)
                st.session_state["diag_report"] = diag_df

        # Show cached report if available
        cached_report = st.session_state.get("diag_report")
        if cached_report is not None and not cached_report.empty:
            st.dataframe(cached_report, width="stretch", hide_index=True)
