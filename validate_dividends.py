#!/usr/bin/env python3
"""
validate_dividends.py — Sanity-check script for dividend implementation.

Compares SPY backtested total return over a 5-year period under three
scenarios:
  a. Price return only (no dividends)
  b. Total return with distributions accrued to cash (not reinvested)
  c. Total return with full DRIP reinvestment

Also validates dividend frequency detection.
"""

import datetime as dt
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Minimal Streamlit stub for caching when running outside Streamlit
import streamlit as st  # noqa: E402

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from data_fetcher import (  # noqa: E402
    fetch_multiple_histories,
    fetch_dividend_history,
    fetch_total_returns,
    fetch_returns,
    detect_distribution_frequency,
    compute_standardized_yield,
    get_dividend_events,
)


def _cagr(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / years) - 1


def run_spy_validation():
    """Run 5-year SPY backtest under three scenarios."""
    ticker = "SPY"
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=5 * 365)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    print("=" * 70)
    print("DIVIDEND IMPLEMENTATION VALIDATION — SPY 5-Year Backtest")
    print("=" * 70)
    print(f"Period: {start_str} to {end_str}")
    print()

    # Fetch price data
    prices_df = fetch_multiple_histories([ticker], start=start_str, end=end_str)
    if prices_df.empty or ticker not in prices_df.columns:
        print("ERROR: Could not fetch SPY price data")
        return False

    prices = prices_df[ticker]
    start_price = float(prices.iloc[0])
    end_price = float(prices.iloc[-1])
    n_years = len(prices) / 252

    # Fetch dividend data
    divs = fetch_dividend_history(ticker, start=start_str, end=end_str)

    print(f"Start price: ${start_price:.2f}")
    print(f"End price:   ${end_price:.2f}")
    print(f"Trading days: {len(prices)}")
    print(f"Years: {n_years:.2f}")
    print(f"Dividends found: {len(divs)}")
    if not divs.empty:
        print(f"Total dividends per share: ${divs.sum():.4f}")
    print()

    # ===== Scenario A: Price return only (no dividends) =====
    price_return = (end_price / start_price) - 1
    price_cagr = _cagr(start_price, end_price, n_years)

    print("--- Scenario A: Price Return Only ---")
    print(f"  Cumulative return: {price_return:.2%}")
    print(f"  CAGR:              {price_cagr:.2%}")
    print()

    # ===== Scenario B: Total return, dividends to cash (not reinvested) =====
    total_divs_ps = divs.sum() if not divs.empty else 0.0
    # Start with 1 share, collect dividends, end value = end_price + total_divs
    total_return_cash = ((end_price + total_divs_ps) / start_price) - 1
    total_cagr_cash = _cagr(start_price, end_price + total_divs_ps, n_years)

    print("--- Scenario B: Total Return, Dividends to Cash ---")
    print(f"  Cumulative return: {total_return_cash:.2%}")
    print(f"  CAGR:              {total_cagr_cash:.2%}")
    print()

    # ===== Scenario C: Total return with DRIP reinvestment =====
    # Simulate DRIP: start with $start_price invested
    shares = 1.0
    for ex_ts, div_amount in divs.items():
        ex_date = ex_ts.date() if hasattr(ex_ts, 'date') else ex_ts
        # Find price on ex-date (or nearest)
        ex_ts_norm = pd.Timestamp(ex_date)
        if ex_ts_norm in prices.index:
            ex_price = float(prices.loc[ex_ts_norm])
        else:
            # Find nearest date
            idx = prices.index.searchsorted(ex_ts_norm)
            idx = min(idx, len(prices) - 1)
            ex_price = float(prices.iloc[idx])

        if ex_price > 0:
            # Dividend received
            div_cash = div_amount * shares
            # DRIP: buy more shares at ex-date price (simplified; real DRIP
            # uses payment date price, but for validation this is close enough)
            new_shares = div_cash / ex_price
            shares += new_shares

    drip_end_value = shares * end_price
    drip_return = (drip_end_value / start_price) - 1
    drip_cagr = _cagr(start_price, drip_end_value, n_years)

    print("--- Scenario C: Total Return with DRIP ---")
    print(f"  Final shares: {shares:.6f} (started with 1.000000)")
    print(f"  Final value:  ${drip_end_value:.2f} (started at ${start_price:.2f})")
    print(f"  Cumulative return: {drip_return:.2%}")
    print(f"  CAGR:              {drip_cagr:.2%}")
    print()

    # ===== Summary =====
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<35} {'Cum Return':>12} {'CAGR':>10}")
    print("-" * 57)
    print(f"{'A: Price Only':<35} {price_return:>12.2%} {price_cagr:>10.2%}")
    print(f"{'B: Total Return (Cash)':<35} {total_return_cash:>12.2%} {total_cagr_cash:>10.2%}")
    print(f"{'C: Total Return (DRIP)':<35} {drip_return:>12.2%} {drip_cagr:>10.2%}")
    print()

    diff_ba = total_return_cash - price_return
    diff_ca = drip_return - price_return
    diff_cb = drip_return - total_return_cash
    print(f"Dividend impact (B - A): {diff_ba:+.2%} cumulative")
    print(f"Dividend impact (C - A): {diff_ca:+.2%} cumulative")
    print(f"DRIP compounding (C - B): {diff_cb:+.2%} cumulative")
    print()

    # Validate results make sense
    ok = True
    if diff_ba < 0.01:
        print("WARNING: Dividend impact seems too small (< 1% over 5 years)")
        ok = False
    if diff_ca <= diff_ba:
        print("WARNING: DRIP should outperform cash accrual due to compounding")
        ok = False
    if total_divs_ps <= 0:
        print("WARNING: No dividends found for SPY")
        ok = False

    if ok:
        print("PASS: All three scenarios show meaningfully different results.")
    else:
        print("CHECK: Some results may need investigation.")

    return ok


def run_frequency_detection():
    """Test frequency detection for common ETFs."""
    print()
    print("=" * 70)
    print("DISTRIBUTION FREQUENCY DETECTION")
    print("=" * 70)

    test_tickers = ["SPY", "AGG"]
    expected = {
        "SPY": "Quarterly",
        "AGG": "Monthly",
    }

    for ticker in test_tickers:
        divs = fetch_dividend_history(ticker)
        if divs.empty:
            print(f"  {ticker}: No dividend data available")
            continue
        freq_label, dpy = detect_distribution_frequency(divs)
        y, _, _ = compute_standardized_yield(ticker)
        exp = expected.get(ticker, "Unknown")
        status = "PASS" if freq_label == exp else "CHECK"
        print(f"  {ticker}: {freq_label} ({dpy}x/yr), yield={y:.2%}  "
              f"[expected {exp}] — {status}")

    print()


def run_total_returns_validation():
    """Compare price-only vs total returns from data_fetcher."""
    print("=" * 70)
    print("TOTAL RETURNS vs PRICE-ONLY RETURNS")
    print("=" * 70)

    start = (dt.date.today() - dt.timedelta(days=365 * 3)).isoformat()
    tickers = ["SPY"]

    price_rets = fetch_returns(tickers, start=start)
    total_rets = fetch_total_returns(tickers, start=start)

    if price_rets.empty or total_rets.empty:
        print("  Could not fetch return data")
        return

    if "SPY" in price_rets.columns and "SPY" in total_rets.columns:
        pr_ann = float(price_rets["SPY"].mean() * 252)
        tr_ann = float(total_rets["SPY"].mean() * 252)
        diff = tr_ann - pr_ann
        print(f"  Price-only annualized return: {pr_ann:.2%}")
        print(f"  Total annualized return:      {tr_ann:.2%}")
        print(f"  Dividend contribution:        {diff:+.2%}")
        if diff > 0:
            print("  PASS: Total returns exceed price returns")
        else:
            print("  CHECK: Total returns should exceed price returns")
    print()


if __name__ == "__main__":
    run_spy_validation()
    run_frequency_detection()
    run_total_returns_validation()
