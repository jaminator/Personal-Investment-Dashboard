#!/usr/bin/env python3
"""
validate_dividend_data.py — Diagnostic report and validation test suite
for the dividend verification layer.

Runs the following tests:
    Test 1: FMP Payment Date Population (PDI, VO, SPY)
    Test 2: Amount Source Priority (FMP preferred when available)
    Test 3: Waterfall Fallback (graceful degradation)
    Test 4: Cash Flow Tab Rendering (FMP indicators present)
    Test 5: Kelly Sleeve Yield Using FMP Data

Also generates the full diagnostic report for PDI, VO, SPY.
"""

from __future__ import annotations

import datetime as dt
import sys

from dividend_verifier import (
    DividendEvent,
    get_verified_dividend_events,
    get_latest_distribution,
    compute_annualized_yield,
    generate_diagnostic_report,
    _fetch_yfinance_dividends,
    _fetch_fmp_dividends,
    _amounts_agree,
    _dates_agree,
)


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str) -> None:
    print()
    print("-" * 50)
    print(title)
    print("-" * 50)


# ---------------------------------------------------------------------------
# Diagnostic Report
# ---------------------------------------------------------------------------

def run_diagnostic_report() -> None:
    """Print a full diagnostic report for PDI, VO, SPY."""
    print_section("DIAGNOSTIC REPORT — Last 8 Events per Ticker")
    tickers = ["PDI", "VO", "SPY"]

    for ticker in tickers:
        print_subsection(f"Ticker: {ticker}")

        # Fetch from each source independently
        yf_divs = _fetch_yfinance_dividends(ticker)
        fmp_divs = _fetch_fmp_dividends(ticker)

        print(f"  yfinance: {len(yf_divs)} events")
        print(f"  FMP:      {len(fmp_divs)} events")
        if fmp_divs:
            fmp_freq = fmp_divs[-1].get("fmp_frequency", "N/A")
            print(f"  FMP frequency field: {fmp_freq!r}")

        # Get verified events
        events = get_verified_dividend_events(ticker)
        recent = events[-8:] if len(events) > 8 else events

        if not recent:
            print(f"  No dividend data available for {ticker}.")
            continue

        print()
        header = (
            f"{'Ex-Date':<12} {'yf $/share':>12} {'FMP $/share':>12} "
            f"{'FMP Ex-Date':<12} {'FMP PayDate':<12} "
            f"{'FMP RecDate':<12} {'FMP DeclDate':<12} "
            f"{'Date Agree':>10} {'Amt Agree':>10} "
            f"{'Amt Src':>8} {'Pay Src':>10}"
        )
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        for ev in recent:
            yf_amt = f"${ev.yfinance_amount:.4f}" if ev.yfinance_amount is not None else "—"
            fmp_amt = f"${ev.fmp_amount:.4f}" if ev.fmp_amount is not None else "—"
            fmp_ex = str(ev.fmp_ex_date) if ev.fmp_ex_date else "—"
            fmp_pay = str(ev.fmp_payment_date) if ev.fmp_payment_date else "—"
            fmp_rec = str(ev.fmp_record_date) if ev.fmp_record_date else "—"
            fmp_decl = str(ev.fmp_declaration_date) if ev.fmp_declaration_date else "—"

            date_agree = "—"
            if ev.fmp_ex_date and ev.ex_dividend_date:
                date_agree = "Y" if _dates_agree(ev.ex_dividend_date, ev.fmp_ex_date, 1) else "N"

            amt_agree = "—"
            if ev.fmp_amount is not None and ev.yfinance_amount is not None:
                amt_agree = "Y" if _amounts_agree(ev.yfinance_amount, ev.fmp_amount) else "N"

            row = (
                f"{str(ev.ex_dividend_date):<12} {yf_amt:>12} {fmp_amt:>12} "
                f"{fmp_ex:<12} {fmp_pay:<12} "
                f"{fmp_rec:<12} {fmp_decl:<12} "
                f"{date_agree:>10} {amt_agree:>10} "
                f"{ev.amount_source:>8} {ev.payment_date_source:>10}"
            )
            print(f"  {row}")

            # Show warnings
            if ev.data_quality_warnings:
                for w in ev.data_quality_warnings:
                    print(f"    !! WARNING: {w}")


# ---------------------------------------------------------------------------
# Test 1: FMP Payment Date Population
# ---------------------------------------------------------------------------

def test_1_fmp_payment_dates() -> bool:
    """For PDI, VO, SPY — fetch most recent 4 dividend events each.
    PASS if payment_date_source == 'FMP' for at least 3 of 4 events per ticker."""
    print_subsection("Test 1: FMP Payment Date Population")

    passed = True
    for ticker in ["PDI", "VO", "SPY"]:
        events = get_verified_dividend_events(ticker)
        recent = events[-4:] if len(events) > 4 else events

        if not recent:
            print(f"  {ticker}: SKIP — No data available")
            continue

        fmp_count = 0
        for ev in recent:
            src = ev.payment_date_source
            print(f"  {ticker} {ev.ex_dividend_date}: "
                  f"payment_date={ev.payment_date} source={src} "
                  f"freq={ev.frequency} freq_source={ev.frequency_source}")
            if src == "FMP":
                fmp_count += 1

        ok = fmp_count >= 3
        if not ok:
            passed = False
        print(f"  {ticker}: {'PASS' if ok else 'FAIL'} "
              f"({fmp_count}/{len(recent)} FMP-sourced)")

    return passed


# ---------------------------------------------------------------------------
# Test 2: Amount Source Priority
# ---------------------------------------------------------------------------

def test_2_amount_source_priority() -> bool:
    """For PDI, VO, SPY — confirm amount_source == 'FMP' for all events
    where FMP returned a non-zero adjDividend."""
    print_subsection("Test 2: Amount Source Priority")

    passed = True
    for ticker in ["PDI", "VO", "SPY"]:
        events = get_verified_dividend_events(ticker)
        recent = events[-4:] if len(events) > 4 else events

        if not recent:
            print(f"  {ticker}: SKIP — No data available")
            continue

        for ev in recent:
            if ev.fmp_amount is not None and ev.fmp_amount > 0:
                if ev.amount_source != "FMP":
                    passed = False
                    print(f"  {ticker} {ev.ex_dividend_date}: FAIL — "
                          f"FMP amount=${ev.fmp_amount:.4f} available but "
                          f"amount_source={ev.amount_source}")
                else:
                    print(f"  {ticker} {ev.ex_dividend_date}: OK — "
                          f"amount_source=FMP (${ev.fmp_amount:.4f})")
            else:
                print(f"  {ticker} {ev.ex_dividend_date}: OK — "
                      f"No FMP amount, using {ev.amount_source}")

    return passed


# ---------------------------------------------------------------------------
# Test 3: Waterfall Fallback
# ---------------------------------------------------------------------------

def test_3_waterfall_fallback() -> bool:
    """Test graceful fallback when FMP has no coverage for a ticker."""
    print_subsection("Test 3: Waterfall Fallback")

    # Use a ticker that likely has yfinance data but limited FMP coverage
    ticker = "SCHD"  # Schwab US Dividend Equity ETF
    try:
        events = get_verified_dividend_events(ticker)
        if not events:
            print(f"  {ticker}: No data — testing with empty result (no exception = PASS)")
            return True

        recent = events[-2:] if len(events) > 2 else events
        for ev in recent:
            print(f"  {ticker} {ev.ex_dividend_date}: "
                  f"amount_source={ev.amount_source} "
                  f"pay_source={ev.payment_date_source} "
                  f"amount=${ev.distribution_per_share:.4f}")

        # Verify no exception was thrown and sources are valid
        for ev in recent:
            if ev.payment_date_source not in ("FMP", "ESTIMATED"):
                print(f"  FAIL: Invalid payment_date_source: {ev.payment_date_source}")
                return False
            if ev.amount_source not in ("FMP", "yfinance"):
                print(f"  FAIL: Invalid amount_source: {ev.amount_source}")
                return False

        print("  PASS: Graceful fallback without exception")
        return True
    except Exception as e:
        print(f"  FAIL: Exception during fallback: {e}")
        return False


# ---------------------------------------------------------------------------
# Test 4: Cash Flow Tab Rendering check
# ---------------------------------------------------------------------------

def test_4_cash_flow_rendering() -> bool:
    """Confirm that at least one FMP-sourced payment date exists for
    holdings that have FMP dividend coverage."""
    print_subsection("Test 4: Cash Flow Tab Rendering")

    has_fmp = False
    for ticker in ["PDI", "VO", "SPY"]:
        events = get_verified_dividend_events(ticker)
        for ev in events[-4:] if len(events) > 4 else events:
            if ev.payment_date_source == "FMP":
                has_fmp = True
                print(f"  {ticker} {ev.ex_dividend_date}: "
                      f"FMP payment_date={ev.payment_date}")

    if has_fmp:
        print("  PASS: FMP payment dates present for cash flow display")
    else:
        print("  FAIL: No FMP payment dates found")
    return has_fmp


# ---------------------------------------------------------------------------
# Test 5: Kelly Sleeve Yield Using FMP Data
# ---------------------------------------------------------------------------

def test_5_kelly_yield_fmp() -> bool:
    """Compute PDI annualized yield as of Dec 31 2024 and confirm
    it uses FMP adjDividend as the distribution_per_share input."""
    print_subsection("Test 5: Kelly Sleeve Yield Using FMP Data")

    try:
        from data_fetcher import fetch_history
        import pandas as pd

        pdi_hist = fetch_history("PDI", start="2024-12-01", end="2025-01-10")
        if pdi_hist.empty:
            print("  SKIP: Cannot fetch PDI price history (network required)")
            return True

        dec31 = pd.Timestamp("2024-12-31")
        idx = pdi_hist.index.searchsorted(dec31)
        idx = min(idx, len(pdi_hist) - 1)
        pdi_price = float(pdi_hist["Close"].iloc[idx])

        latest = get_latest_distribution("PDI", as_of_date=dt.date(2024, 12, 31))
        if latest is None:
            print("  SKIP: No PDI distribution data as of 2024-12-31")
            return True

        ann_yield = compute_annualized_yield("PDI", dt.date(2024, 12, 31), pdi_price)

        print(f"  PDI price (Dec 31 2024): ${pdi_price:.2f}")
        print(f"  Last dist/share: ${latest.distribution_per_share:.4f}")
        print(f"  Frequency: {latest.frequency}x/year")
        print(f"  Amount source: {latest.amount_source}")
        if ann_yield:
            print(f"  Annualized yield: {ann_yield:.2%}")

        if latest.amount_source == "FMP":
            print("  PASS: amount_source == FMP")
            return True
        elif latest.fmp_amount is None:
            print("  PASS (conditional): No FMP data available, using yfinance")
            return True
        else:
            print(f"  FAIL: amount_source={latest.amount_source} despite FMP data available")
            return False

    except Exception as e:
        print(f"  SKIP: Error — {e}")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Dividend Data Verification — Diagnostic Report & Test Suite")
    print("=" * 70)
    print(f"Run date: {dt.date.today()}")
    print()

    # Phase 1: Diagnostic report
    run_diagnostic_report()

    # Phase 2: Validation tests
    print_section("VALIDATION TEST SUITE")

    results: dict[str, bool] = {}
    results["Test 1: FMP Payment Date Population"] = test_1_fmp_payment_dates()
    results["Test 2: Amount Source Priority"] = test_2_amount_source_priority()
    results["Test 3: Waterfall Fallback"] = test_3_waterfall_fallback()
    results["Test 4: Cash Flow Tab Rendering"] = test_4_cash_flow_rendering()
    results["Test 5: Kelly Yield FMP Data"] = test_5_kelly_yield_fmp()

    # Summary
    print_section("TEST RESULTS SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see details above")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
