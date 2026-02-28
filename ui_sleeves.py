"""
ui_sleeves.py — Sleeve Management UI (Feature 5).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_fetcher import compute_standardized_yield
from dividend_verifier import compute_annualized_yield as verified_yield
from portfolio import (
    FREQUENCIES,
    REBALANCE_MODES,
    PortfolioConfig,
    Sleeve,
)


def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def _pct(val: float) -> str:
    return f"{val:.2%}"


def render_sleeves(config: PortfolioConfig) -> None:
    """Render the sleeve management panel."""
    st.header("Sleeve Manager")

    # --- Summary table of sleeves -------------------------------------------
    if config.sleeves:
        st.subheader("Current Sleeves")
        total_val = config.total_value
        rows = []
        for s in config.sleeves:
            sleeve_val = sum(
                (config.holding_by_id(hid).market_value if config.holding_by_id(hid) else 0)
                for hid in s.holding_ids
            ) + s.cash_balance
            holdings_str = ", ".join(
                config.holding_by_id(hid).ticker
                for hid in s.holding_ids
                if config.holding_by_id(hid) and config.holding_by_id(hid).ticker
            )
            rows.append({
                "Sleeve Name": s.name,
                "Holdings": holdings_str or "—",
                "Current Value": _fmt(sleeve_val),
                "% of Portfolio": _pct(sleeve_val / total_val) if total_val > 0 else "0.00%",
                "Rebalancing Mode": s.mode,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- Cross-sleeve toggle ------------------------------------------------
    config.cross_sleeve_cash_pooling = st.checkbox(
        "Enable cross-sleeve cash pooling",
        value=config.cross_sleeve_cash_pooling,
        help="When enabled, cash from SELL trades can flow between sleeves.",
    )

    # --- Create new sleeve --------------------------------------------------
    st.subheader("Create New Sleeve")
    with st.form("create_sleeve_form", clear_on_submit=True):
        name = st.text_input("Sleeve Name", placeholder="e.g. Core Equity")
        mode = st.selectbox("Rebalancing Mode", REBALANCE_MODES)

        # Holdings multi-select
        all_holdings = {
            f"{h.ticker or 'Manual'} ({h.id})": h.id
            for h in config.holdings
        }
        assigned = set()
        for s in config.sleeves:
            assigned.update(s.holding_ids)
        available = {k: v for k, v in all_holdings.items() if v not in assigned}

        selected = st.multiselect("Assign Holdings", list(available.keys()))
        selected_ids = [available[k] for k in selected]

        submitted = st.form_submit_button("Create Sleeve")
        if submitted and name:
            new_sleeve = Sleeve(name=name, mode=mode, holding_ids=selected_ids)
            config.sleeves.append(new_sleeve)
            st.rerun()

    # --- Edit / configure each sleeve ---------------------------------------
    if config.sleeves:
        st.subheader("Configure Sleeves")
        for s in config.sleeves:
            with st.expander(f"Sleeve: {s.name} ({s.mode})", expanded=False):
                _render_sleeve_config(config, s)

    # --- Unassigned holdings ------------------------------------------------
    unassigned = config.unassigned_holdings()
    if unassigned:
        st.subheader("Unmanaged Holdings")
        st.caption("These holdings are not assigned to any sleeve and will not be rebalanced.")
        for h in unassigned:
            st.write(f"- {h.ticker or 'Manual'}: {_fmt(h.market_value)}")


def _render_sleeve_config(config: PortfolioConfig, sleeve: Sleeve) -> None:
    """Render configuration controls for a single sleeve."""
    key_prefix = f"sleeve_{sleeve.id}"

    # --- Mode A: Target-Weight ---------------------------------------------
    if sleeve.mode.startswith("A"):
        st.markdown("**Target-Weight Rebalancing**")

        # Target weights per holding
        for hid in sleeve.holding_ids:
            h = config.holding_by_id(hid)
            if not h:
                continue
            label = h.ticker or "Manual"
            current = sleeve.target_weights.get(hid, 0.0)
            new_w = st.number_input(
                f"Target weight for {label} (%)",
                min_value=0.0, max_value=100.0,
                value=current,
                step=1.0,
                key=f"{key_prefix}_tw_{hid}",
            )
            sleeve.target_weights[hid] = new_w

        total_w = sum(sleeve.target_weights.get(hid, 0.0) for hid in sleeve.holding_ids)
        if total_w > 0 and abs(total_w - 100.0) > 0.1:
            st.warning(f"Target weights sum to {total_w:.1f}% (should be 100%).")

        sleeve.rebalance_trigger = st.selectbox(
            "Rebalancing Trigger",
            ["Monthly", "Quarterly", "Semi-Annually", "Annually", "Threshold", "Hybrid"],
            index=["Monthly", "Quarterly", "Semi-Annually", "Annually", "Threshold", "Hybrid"].index(
                sleeve.rebalance_trigger
            ) if sleeve.rebalance_trigger in ["Monthly", "Quarterly", "Semi-Annually", "Annually", "Threshold", "Hybrid"] else 1,
            key=f"{key_prefix}_trigger",
        )

        if sleeve.rebalance_trigger in ("Threshold", "Hybrid"):
            sleeve.threshold_pct = st.number_input(
                "Drift threshold (%)",
                min_value=0.1, max_value=50.0,
                value=sleeve.threshold_pct,
                step=0.5,
                key=f"{key_prefix}_threshold",
            )

        sleeve.rebalance_method = st.selectbox(
            "Rebalancing Method",
            ["Full", "Contributions-only", "Partial"],
            index=["Full", "Contributions-only", "Partial"].index(sleeve.rebalance_method)
            if sleeve.rebalance_method in ["Full", "Contributions-only", "Partial"] else 0,
            key=f"{key_prefix}_method",
        )

        sleeve.transaction_cost_pct = st.number_input(
            "Transaction cost (%)",
            min_value=0.0, max_value=5.0,
            value=sleeve.transaction_cost_pct,
            step=0.01,
            key=f"{key_prefix}_txcost",
        )

    # --- Mode B: Signal-Based ---------------------------------------------
    elif sleeve.mode.startswith("B"):
        st.markdown("**Signal-Based Rebalancing (Kelly 3% Rule)**")

        if len(sleeve.holding_ids) != 2:
            st.warning("Mode B requires exactly 2 holdings assigned to this sleeve.")

        holding_tickers = []
        for hid in sleeve.holding_ids:
            h = config.holding_by_id(hid)
            if h and h.ticker:
                holding_tickers.append(h.ticker)

        if len(holding_tickers) >= 2:
            sleeve.bond_ticker = st.selectbox(
                "Bond / Income instrument",
                holding_tickers,
                index=holding_tickers.index(sleeve.bond_ticker) if sleeve.bond_ticker in holding_tickers else 0,
                key=f"{key_prefix}_bond",
            )
            remaining = [t for t in holding_tickers if t != sleeve.bond_ticker]
            sleeve.stock_ticker = st.selectbox(
                "Growth / Equity instrument",
                remaining,
                index=0,
                key=f"{key_prefix}_stock",
            )
        else:
            sleeve.bond_ticker = st.text_input("Bond Ticker", value=sleeve.bond_ticker, key=f"{key_prefix}_bond_txt")
            sleeve.stock_ticker = st.text_input("Stock Ticker", value=sleeve.stock_ticker, key=f"{key_prefix}_stock_txt")

        sleeve.annualized_yield_pct = st.number_input(
            "Annualized yield target (%)",
            min_value=0.0, max_value=100.0,
            value=sleeve.annualized_yield_pct,
            step=0.5,
            key=f"{key_prefix}_yield",
        )

        # Show actual trailing yields for assigned holdings (verified)
        if holding_tickers:
            yield_info = []
            for ht in holding_tickers:
                vy = verified_yield(ht)
                y, dpy, freq = compute_standardized_yield(ht)
                if vy is not None and vy > 0:
                    yield_info.append(f"{ht}: {vy:.2%} ({freq}, verified)")
                elif y > 0:
                    yield_info.append(f"{ht}: {y:.2%} ({freq})")
                else:
                    yield_info.append(f"{ht}: no yield data")
            st.caption("Actual TTM yields: " + " | ".join(yield_info))

        sleeve.signal_frequency = st.selectbox(
            "Rebalancing Frequency",
            FREQUENCIES,
            index=FREQUENCIES.index(sleeve.signal_frequency) if sleeve.signal_frequency in FREQUENCIES else 4,
            key=f"{key_prefix}_freq",
        )

        sleeve.signal_scaling = st.number_input(
            "Signal scaling factor",
            min_value=0.01, max_value=10.0,
            value=sleeve.signal_scaling,
            step=0.1,
            key=f"{key_prefix}_scaling",
        )

        sleeve.force_sideways_trade = st.checkbox(
            "Force trade in sideways market",
            value=sleeve.force_sideways_trade,
            key=f"{key_prefix}_force",
        )

        sleeve.transaction_cost_pct = st.number_input(
            "Transaction cost (%)",
            min_value=0.0, max_value=5.0,
            value=sleeve.transaction_cost_pct,
            step=0.01,
            key=f"{key_prefix}_txcost_b",
        )

    # --- Mode C: Custom Formula -------------------------------------------
    elif sleeve.mode.startswith("C"):
        st.markdown("**Custom Formula Rebalancing**")
        st.caption(
            "Enter a Python expression. Available variables: "
            "`portfolio[\"TICKER\"].value`, `.price`, `.shares`, "
            "`.yield_ttm` (real trailing distribution yield from yfinance); "
            "`portfolio.total_value`; `date`; `cash`. "
            "Positive result → BUY first holding, SELL second. "
            "Negative → reverse."
        )

        # Show current yield_ttm values for assigned tickers
        formula_tickers = [
            config.holding_by_id(hid).ticker
            for hid in sleeve.holding_ids
            if config.holding_by_id(hid) and config.holding_by_id(hid).ticker
        ]
        if formula_tickers:
            yield_info = []
            for ft in formula_tickers:
                vy = verified_yield(ft)
                y, dpy, freq = compute_standardized_yield(ft)
                effective = vy if vy is not None and vy > 0 else y
                verified_tag = " (verified)" if vy is not None and vy > 0 else ""
                yield_info.append(f"{ft}.yield_ttm = {effective:.4f} ({freq}{verified_tag})")
            st.info("Current yield values: " + " | ".join(yield_info))
        sleeve.custom_formula = st.text_area(
            "Formula",
            value=sleeve.custom_formula,
            height=100,
            key=f"{key_prefix}_formula",
        )

        sleeve.transaction_cost_pct = st.number_input(
            "Transaction cost (%)",
            min_value=0.0, max_value=5.0,
            value=sleeve.transaction_cost_pct,
            step=0.01,
            key=f"{key_prefix}_txcost_c",
        )

    # --- Delete sleeve ------------------------------------------------------
    if st.button(f"Delete sleeve '{sleeve.name}'", key=f"{key_prefix}_delete"):
        config.sleeves = [s for s in config.sleeves if s.id != sleeve.id]
        st.rerun()
