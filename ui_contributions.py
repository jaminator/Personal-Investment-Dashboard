"""
ui_contributions.py — Recurring Contributions & Discrete Withdrawals UI (Features 2 & 3).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from contributions import build_contribution_schedule, build_withdrawal_schedule
from portfolio import FREQUENCIES, ContributionStream, PortfolioConfig, Withdrawal


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def render_contributions(config: PortfolioConfig) -> None:
    """Render the contributions and withdrawals management panel."""
    st.header("Cash Flows")

    tab_contrib, tab_withdraw = st.tabs(["Recurring Contributions", "Discrete Withdrawals"])

    # ======================================================================
    # Contributions tab
    # ======================================================================
    with tab_contrib:
        st.subheader("Add Contribution Stream")
        with st.form("add_contrib_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=500.0, step=50.0)
            with c2:
                frequency = st.selectbox("Frequency", FREQUENCIES, index=4)  # Monthly
            with c3:
                alloc_mode = st.selectbox(
                    "Allocation",
                    ["Proportional", "Specific Ticker", "Cash"],
                )
            c4, c5, c6 = st.columns(3)
            with c4:
                start_date = st.date_input("Start Date", value=dt.date.today())
            with c5:
                use_end = st.checkbox("Set end date")
            with c6:
                end_date = st.date_input(
                    "End Date",
                    value=dt.date.today() + dt.timedelta(days=365 * 5),
                    disabled=not use_end,
                )

            target_ticker = ""
            target_sleeve = ""
            if alloc_mode == "Specific Ticker":
                target_ticker = st.text_input("Target Ticker", placeholder="e.g. SPY").upper().strip()

            sleeve_names = [s.name for s in config.sleeves]
            if sleeve_names:
                target_sleeve = st.selectbox(
                    "Target Sleeve (optional)",
                    ["Proportional across sleeves"] + sleeve_names,
                )
                if target_sleeve == "Proportional across sleeves":
                    target_sleeve = ""

            submitted = st.form_submit_button("Add Contribution")
            if submitted:
                cs = ContributionStream(
                    amount=amount,
                    frequency=frequency,
                    start_date=str(start_date),
                    end_date=str(end_date) if use_end else "",
                    allocation_mode=alloc_mode,
                    target_ticker=target_ticker,
                    target_sleeve=target_sleeve,
                )
                config.contributions.append(cs)
                st.rerun()

        # --- Current contribution streams -----------------------------------
        if config.contributions:
            st.subheader("Active Contribution Streams")
            rows = []
            for cs in config.contributions:
                rows.append({
                    "ID": cs.id,
                    "Amount": _fmt(cs.amount),
                    "Frequency": cs.frequency,
                    "Start": cs.start_date or "Portfolio start",
                    "End": cs.end_date or "No end",
                    "Allocation": cs.allocation_mode,
                    "Target": cs.target_ticker or cs.target_sleeve or "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["ID"]), width="stretch", hide_index=True)

            # Remove
            remove_opts = {
                f"{_fmt(cs.amount)} {cs.frequency} ({cs.id})": cs.id
                for cs in config.contributions
            }
            to_remove = st.multiselect("Remove contributions", list(remove_opts.keys()))
            if st.button("Remove Selected Contributions") and to_remove:
                ids = {remove_opts[k] for k in to_remove}
                config.contributions = [c for c in config.contributions if c.id not in ids]
                st.rerun()

        # --- Projected schedule ---------------------------------------------
        if config.contributions:
            st.subheader("Projected Contribution Schedule")
            forecast_end = dt.date.today() + dt.timedelta(days=365 * config.forecast_years)
            sched = build_contribution_schedule(config.contributions, dt.date.today(), forecast_end)
            if not sched.empty:
                total = sched["amount"].sum()
                st.metric("Total Projected Contributions", _fmt(total))
                # Show first 50 rows
                display = sched[["date", "amount"]].copy()
                display["amount"] = display["amount"].apply(_fmt)
                st.dataframe(display.head(50), width="stretch", hide_index=True)
                if len(sched) > 50:
                    st.caption(f"Showing first 50 of {len(sched)} scheduled contributions.")
            else:
                st.info("No contributions scheduled in the forecast period.")

    # ======================================================================
    # Withdrawals tab
    # ======================================================================
    with tab_withdraw:
        st.subheader("Add Withdrawal")
        with st.form("add_withdrawal_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                w_date = st.date_input("Withdrawal Date", value=dt.date.today() + dt.timedelta(days=365))
            with c2:
                w_amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=1000.0)
            with c3:
                is_pct = st.checkbox("Amount is % of portfolio value")

            label = st.text_input("Label / Description", placeholder="e.g. Down payment on house")

            submitted_w = st.form_submit_button("Add Withdrawal")
            if submitted_w:
                w = Withdrawal(
                    date=str(w_date),
                    amount=w_amount,
                    is_percentage=is_pct,
                    label=label,
                )
                config.withdrawals.append(w)
                st.rerun()

        if config.withdrawals:
            st.subheader("Scheduled Withdrawals")
            rows = []
            for w in sorted(config.withdrawals, key=lambda x: x.date):
                rows.append({
                    "ID": w.id,
                    "Date": w.date,
                    "Amount": f"{w.amount:.1f}%" if w.is_percentage else _fmt(w.amount),
                    "Type": "% of Portfolio" if w.is_percentage else "Fixed $",
                    "Label": w.label or "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["ID"]), width="stretch", hide_index=True)

            remove_opts_w = {
                f"{w.label or w.date} ({w.id})": w.id
                for w in config.withdrawals
            }
            to_remove_w = st.multiselect("Remove withdrawals", list(remove_opts_w.keys()))
            if st.button("Remove Selected Withdrawals") and to_remove_w:
                ids = {remove_opts_w[k] for k in to_remove_w}
                config.withdrawals = [w for w in config.withdrawals if w.id not in ids]
                st.rerun()
