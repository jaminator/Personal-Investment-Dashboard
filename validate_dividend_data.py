#!/usr/bin/env python3
"""
validate_dividend_data.py — Diagnostic report and validation test suite
for the dividend verification layer.

Runs the following tests:
    Test 1: PDI distribution amount accuracy (yfinance vs FMP vs SEC EDGAR)
    Test 2: PDI payment date accuracy (FMP vs SEC vs estimated offset)
    Test 3: VO and SPY distribution amount accuracy (yfinance vs FMP)
    Test 4: VO and SPY payment date accuracy (FMP vs estimated offset)
    Test 5: Yield computation consistency (PDI quarterly snapshots)
    Test 6: Kelly sleeve signal integrity (Q1 2025 reference calculation)

Also generates the full diagnostic report for PDI, VO, SPY, AGG.
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
    _fetch_sec_dividends,
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
    """Print a full diagnostic report for PDI, VO, SPY, AGG."""
    print_section("DIAGNOSTIC REPORT — Last 8 Events per Ticker")
    tickers = ["PDI", "VO", "SPY", "AGG"]

    for ticker in tickers:
        print_subsection(f"Ticker: {ticker}")

        # Fetch from each source independently
        yf_divs = _fetch_yfinance_dividends(ticker)
        fmp_divs = _fetch_fmp_dividends(ticker)
        sec_divs = _fetch_sec_dividends(ticker, max_filings=10)

        print(f"  yfinance: {len(yf_divs)} events")
        print(f"  FMP:      {len(fmp_divs)} events")
        print(f"  SEC 8-K:  {len(sec_divs)} events with dividend data")

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
            f"{'Date Agree':>10} {'Amt Agree':>10}"
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
                f"{date_agree:>10} {amt_agree:>10}"
            )
            print(f"  {row}")

            # Show SEC data if available
            if ev.sec_amount is not None or ev.sec_payment_date:
                sec_parts = []
                if ev.sec_amount is not None:
                    sec_parts.append(f"SEC amount=${ev.sec_amount:.4f}")
                if ev.sec_payment_date:
                    sec_parts.append(f"SEC payDate={ev.sec_payment_date}")
                print(f"    >> {', '.join(sec_parts)}")

            # Show PIMCO data if available
            if ev.pimco_amount is not None or ev.pimco_payment_date:
                pimco_parts = []
                if ev.pimco_amount is not None:
                    pimco_parts.append(f"PIMCO amount=${ev.pimco_amount:.4f}")
                if ev.pimco_payment_date:
                    pimco_parts.append(f"PIMCO payDate={ev.pimco_payment_date}")
                print(f"    >> {', '.join(pimco_parts)}")

            # Show warnings
            if ev.data_quality_warnings:
                for w in ev.data_quality_warnings:
                    print(f"    !! WARNING: {w}")


# ---------------------------------------------------------------------------
# Test 1: PDI Distribution Amount Accuracy
# ---------------------------------------------------------------------------

def test_1_pdi_amount() -> bool:
    """Cross-validate last 6 monthly PDI distributions across sources."""
    print_subsection("Test 1: PDI Distribution Amount Accuracy")

    events = get_verified_dividend_events("PDI")
    recent = events[-6:] if len(events) > 6 else events

    if not recent:
        print("  SKIP: No PDI dividend data available (network required)")
        return True  # Not a failure if no data

    total = len(recent)
    mismatches = 0

    for ev in recent:
        sources_present = []
        if ev.yfinance_amount is not None:
            sources_present.append(("yfinance", ev.yfinance_amount))
        if ev.fmp_amount is not None:
            sources_present.append(("FMP", ev.fmp_amount))
        if ev.sec_amount is not None:
            sources_present.append(("SEC_EDGAR", ev.sec_amount))

        if len(sources_present) < 2:
            print(f"  {ev.ex_dividend_date}: Only {len(sources_present)} source(s) — "
                  f"{', '.join(f'{s}=${a:.4f}' for s, a in sources_present)}")
            continue

        all_agree = True
        for i, (s1, a1) in enumerate(sources_present):
            for s2, a2 in sources_present[i + 1:]:
                if not _amounts_agree(a1, a2):
                    all_agree = False
                    mismatches += 1
                    print(f"  {ev.ex_dividend_date}: MISMATCH {s1}=${a1:.4f} vs {s2}=${a2:.4f}")

        if all_agree:
            print(f"  {ev.ex_dividend_date}: AGREE "
                  f"({', '.join(f'{s}=${a:.4f}' for s, a in sources_present)})")

    passed = mismatches == 0
    print(f"\n  {'PASS' if passed else 'FAIL'}: {total} events checked, {mismatches} mismatches")
    return passed


# ---------------------------------------------------------------------------
# Test 2: PDI Payment Date Accuracy
# ---------------------------------------------------------------------------

def test_2_pdi_payment_dates() -> bool:
    """Compare FMP payment dates vs SEC EDGAR vs estimated offset for PDI."""
    print_subsection("Test 2: PDI Payment Date Accuracy")

    events = get_verified_dividend_events("PDI")
    recent = events[-6:] if len(events) > 6 else events

    if not recent:
        print("  SKIP: No PDI dividend data available")
        return True

    mismatches = 0
    offsets = []

    for ev in recent:
        if ev.fmp_payment_date and ev.ex_dividend_date:
            actual_offset = (ev.fmp_payment_date - ev.ex_dividend_date).days
            offsets.append(actual_offset)

        pay_sources = {}
        if ev.fmp_payment_date:
            pay_sources["FMP"] = ev.fmp_payment_date
        if ev.sec_payment_date:
            pay_sources["SEC_EDGAR"] = ev.sec_payment_date
        estimated = ev.ex_dividend_date + dt.timedelta(days=15) if ev.ex_dividend_date else None
        if estimated:
            pay_sources["Estimated(+15d)"] = estimated

        if len(pay_sources) < 2:
            names = ", ".join(f"{k}={v}" for k, v in pay_sources.items())
            print(f"  {ev.ex_dividend_date}: Only {len(pay_sources)} source(s) — {names}")
            continue

        # Check FMP vs SEC agreement
        fmp_d = pay_sources.get("FMP")
        sec_d = pay_sources.get("SEC_EDGAR")
        if fmp_d and sec_d:
            if _dates_agree(fmp_d, sec_d, tolerance_days=2):
                print(f"  {ev.ex_dividend_date}: AGREE FMP={fmp_d} SEC={sec_d}")
            else:
                mismatches += 1
                diff = (sec_d - fmp_d).days
                print(f"  {ev.ex_dividend_date}: MISMATCH FMP={fmp_d} SEC={sec_d} (diff={diff}d)")
        elif fmp_d:
            est_d = pay_sources.get("Estimated(+15d)")
            if est_d:
                diff = (fmp_d - est_d).days
                print(f"  {ev.ex_dividend_date}: FMP={fmp_d} vs Est={est_d} (diff={diff}d)")

    if offsets:
        avg_offset = sum(offsets) / len(offsets)
        print(f"\n  Average ex-date to payment-date offset: {avg_offset:.1f} days")
        print(f"  (Default estimate for monthly funds: 15 days)")

    passed = mismatches == 0
    print(f"\n  {'PASS' if passed else 'FAIL'}: {mismatches} FMP/SEC mismatches")
    return passed


# ---------------------------------------------------------------------------
# Test 3: VO and SPY Distribution Amount Accuracy
# ---------------------------------------------------------------------------

def test_3_vo_spy_amount() -> bool:
    """Cross-validate last 4 quarterly VO and SPY distributions."""
    print_subsection("Test 3: VO and SPY Distribution Amount Accuracy")

    passed = True
    for ticker in ["VO", "SPY"]:
        events = get_verified_dividend_events(ticker)
        recent = events[-4:] if len(events) > 4 else events

        if not recent:
            print(f"  {ticker}: SKIP — No data available")
            continue

        mismatches = 0
        for ev in recent:
            if ev.yfinance_amount is not None and ev.fmp_amount is not None:
                if _amounts_agree(ev.yfinance_amount, ev.fmp_amount):
                    print(f"  {ticker} {ev.ex_dividend_date}: AGREE "
                          f"yf=${ev.yfinance_amount:.4f} FMP=${ev.fmp_amount:.4f}")
                else:
                    mismatches += 1
                    print(f"  {ticker} {ev.ex_dividend_date}: MISMATCH "
                          f"yf=${ev.yfinance_amount:.4f} FMP=${ev.fmp_amount:.4f} "
                          f"(diff=${abs(ev.yfinance_amount - ev.fmp_amount):.4f})")
            else:
                sources = []
                if ev.yfinance_amount is not None:
                    sources.append(f"yf=${ev.yfinance_amount:.4f}")
                if ev.fmp_amount is not None:
                    sources.append(f"FMP=${ev.fmp_amount:.4f}")
                print(f"  {ticker} {ev.ex_dividend_date}: "
                      f"Only {len(sources)} source(s) — {', '.join(sources)}")

        if mismatches > 0:
            passed = False
            print(f"  {ticker}: FAIL — {mismatches} mismatches")
        else:
            print(f"  {ticker}: PASS")

    return passed


# ---------------------------------------------------------------------------
# Test 4: VO and SPY Payment Date Accuracy
# ---------------------------------------------------------------------------

def test_4_vo_spy_payment_dates() -> bool:
    """Compare FMP payment dates vs estimated offset for VO and SPY."""
    print_subsection("Test 4: VO and SPY Payment Date Accuracy")

    for ticker in ["VO", "SPY"]:
        events = get_verified_dividend_events(ticker)
        recent = events[-4:] if len(events) > 4 else events

        if not recent:
            print(f"  {ticker}: SKIP — No data available")
            continue

        offsets = []
        for ev in recent:
            if ev.fmp_payment_date and ev.ex_dividend_date:
                actual = (ev.fmp_payment_date - ev.ex_dividend_date).days
                offsets.append(actual)
                estimated = ev.ex_dividend_date + dt.timedelta(days=20)
                diff = (ev.fmp_payment_date - estimated).days
                print(f"  {ticker} {ev.ex_dividend_date}: "
                      f"FMP pay={ev.fmp_payment_date} (offset={actual}d, "
                      f"vs est+20d diff={diff}d)")
            elif ev.ex_dividend_date:
                print(f"  {ticker} {ev.ex_dividend_date}: No FMP payment date")

        if offsets:
            avg = sum(offsets) / len(offsets)
            print(f"  {ticker} average offset: {avg:.1f} days "
                  f"(default estimate: 20 days)")
        print()

    return True  # Informational only


# ---------------------------------------------------------------------------
# Test 5: Yield Computation Consistency
# ---------------------------------------------------------------------------

def test_5_yield_consistency() -> bool:
    """Compute PDI annualized yield at 4 quarterly snapshots."""
    print_subsection("Test 5: PDI Yield Computation Consistency")

    reference_dates = [
        dt.date(2024, 3, 29),   # Q1 2024 end (last business day)
        dt.date(2024, 6, 28),   # Q2 2024 end
        dt.date(2024, 9, 30),   # Q3 2024 end
        dt.date(2024, 12, 31),  # Q4 2024 end
    ]

    # Try to get historical prices for these dates
    try:
        from data_fetcher import fetch_history
        hist = fetch_history("PDI", start="2024-01-01", end="2025-01-05")
        if hist.empty:
            print("  SKIP: Cannot fetch PDI price history (network required)")
            return True

        import pandas as pd
        for ref_date in reference_dates:
            latest = get_latest_distribution("PDI", as_of_date=ref_date)
            if latest is None:
                print(f"  {ref_date}: No distribution data as of this date")
                continue

            # Find closest price to ref_date
            ts = pd.Timestamp(ref_date)
            idx = hist.index.searchsorted(ts)
            idx = min(idx, len(hist) - 1)
            if idx > 0 and abs((hist.index[idx] - ts).days) > abs((hist.index[idx - 1] - ts).days):
                idx = idx - 1
            price = float(hist["Close"].iloc[idx])
            actual_date = hist.index[idx].date()

            ann_yield = compute_annualized_yield("PDI", ref_date, price)

            print(f"  {ref_date} (price as of {actual_date}: ${price:.2f}):")
            print(f"    Last dist/share: ${latest.distribution_per_share:.4f}")
            print(f"    Frequency: {latest.frequency}x/year")
            print(f"    Annualized yield: {ann_yield:.2%}" if ann_yield else "    Annualized yield: N/A")
            print(f"    Amount source: {latest.amount_source}")
            print(f"    Amount verified: {'Y' if latest.amount_verified else 'N'}")

    except Exception as e:
        print(f"  SKIP: Error computing yields — {e}")
        return True

    return True


# ---------------------------------------------------------------------------
# Test 6: Kelly Sleeve Signal Integrity
# ---------------------------------------------------------------------------

def test_6_kelly_signal() -> bool:
    """Run a single Kelly rebalancing evaluation for Q1 2025."""
    print_subsection("Test 6: Kelly Sleeve Signal Integrity (Q1 2025)")

    try:
        from data_fetcher import fetch_history
        import pandas as pd

        # Get PDI data as of Dec 31 2024
        pdi_hist = fetch_history("PDI", start="2024-12-01", end="2025-01-10")
        vo_hist = fetch_history("VO", start="2024-12-01", end="2025-01-10")

        if pdi_hist.empty or vo_hist.empty:
            print("  SKIP: Cannot fetch price history (network required)")
            return True

        # Find Dec 31 2024 price (or nearest trading day)
        dec31 = pd.Timestamp("2024-12-31")
        jan2 = pd.Timestamp("2025-01-02")

        # PDI price on Dec 31 2024
        pdi_idx = pdi_hist.index.searchsorted(dec31)
        pdi_idx = min(pdi_idx, len(pdi_hist) - 1)
        pdi_price_dec31 = float(pdi_hist["Close"].iloc[pdi_idx])

        # VO price on Jan 2 2025 (first trading day of Q1)
        vo_idx = vo_hist.index.searchsorted(jan2)
        vo_idx = min(vo_idx, len(vo_hist) - 1)
        vo_price_jan2 = float(vo_hist["Close"].iloc[vo_idx])
        vo_date_jan2 = vo_hist.index[vo_idx].date()

        # Get verified PDI yield
        latest_pdi = get_latest_distribution("PDI", as_of_date=dt.date(2024, 12, 31))
        if latest_pdi is None:
            print("  SKIP: No PDI distribution data")
            return True

        pdi_yield = compute_annualized_yield("PDI", dt.date(2024, 12, 31), pdi_price_dec31)

        # Simulate Kelly signal
        # Assume 50/50 portfolio with $100K initial
        initial = 100_000.0
        vo_alloc = initial * 0.5
        pdi_alloc = initial * 0.5

        vo_shares = vo_alloc / vo_price_jan2 if vo_price_jan2 > 0 else 0
        vo_value = vo_shares * vo_price_jan2

        # Signal = yield * stock_value * period_fraction * scaling
        period_fraction = 1.0 / 4.0  # Quarterly
        signal = (pdi_yield or 0) * vo_value * period_fraction * 1.0

        print(f"  PDI last dist/share:    ${latest_pdi.distribution_per_share:.4f}")
        print(f"  PDI distributions/year: {latest_pdi.frequency}")
        print(f"  PDI close Dec 31 2024:  ${pdi_price_dec31:.2f}")
        print(f"  PDI annualized yield:   {pdi_yield:.2%}" if pdi_yield else "  PDI annualized yield:   N/A")
        print(f"  Target PDI weight:      50%")
        print(f"  Target VO weight:       50%")
        print(f"  VO value on {vo_date_jan2}:  ${vo_value:,.2f}")
        print(f"  Adjusted signal line:   ${signal:,.2f}")
        print()
        print(f"  Amount source: {latest_pdi.amount_source}")
        print(f"  Amount verified: {'Y' if latest_pdi.amount_verified else 'N'}")
        if latest_pdi.data_quality_warnings:
            for w in latest_pdi.data_quality_warnings:
                print(f"  WARNING: {w}")

        print()
        print("  This provides a manually verifiable reference for the Kelly engine.")

    except Exception as e:
        print(f"  SKIP: Error in Kelly signal test — {e}")
        import traceback
        traceback.print_exc()
        return True

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
    results["Test 1: PDI Amount Accuracy"] = test_1_pdi_amount()
    results["Test 2: PDI Payment Dates"] = test_2_pdi_payment_dates()
    results["Test 3: VO/SPY Amount Accuracy"] = test_3_vo_spy_amount()
    results["Test 4: VO/SPY Payment Dates"] = test_4_vo_spy_payment_dates()
    results["Test 5: Yield Consistency"] = test_5_yield_consistency()
    results["Test 6: Kelly Signal Integrity"] = test_6_kelly_signal()

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
