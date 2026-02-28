#!/usr/bin/env python3
"""Validate all fixes from Sections 1-3."""

import datetime as dt
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

passed = 0
failed = 0


def log(status, msg):
    global passed, failed
    if status == "PASS":
        passed += 1
        print(f"  \u2705 PASS: {msg}")
    else:
        failed += 1
        print(f"  \u274c FAIL: {msg}")


# ===========================================================================
# CHECK 1: FMP API key injection pattern (structural)
# ===========================================================================
print("\n=== CHECK 1: FMP API Key Injection Pattern ===")

import inspect
import data_fetcher
import dividend_verifier

# Verify no params={"apikey": ...} in FMP functions
df_src = inspect.getsource(data_fetcher.fetch_fmp_dividends)
if 'params={' not in df_src and 'params =' not in df_src:
    log("PASS", "data_fetcher.fetch_fmp_dividends: no params dict, apikey in URL")
else:
    log("FAIL", "data_fetcher.fetch_fmp_dividends: still uses params dict")

tf_src = inspect.getsource(data_fetcher.test_fmp_connection)
if 'params={' not in tf_src and 'params =' not in tf_src:
    log("PASS", "data_fetcher.test_fmp_connection: no params dict, apikey in URL")
else:
    log("FAIL", "data_fetcher.test_fmp_connection: still uses params dict")

dv_src = inspect.getsource(dividend_verifier._fetch_fmp_dividends)
if 'params={' not in dv_src and 'params =' not in dv_src:
    log("PASS", "dividend_verifier._fetch_fmp_dividends: no params dict, apikey in URL")
else:
    log("FAIL", "dividend_verifier._fetch_fmp_dividends: still uses params dict")

# Verify fetch_fmp_historical_price is removed
if hasattr(data_fetcher, "fetch_fmp_historical_price"):
    log("FAIL", "fetch_fmp_historical_price still exists in data_fetcher")
else:
    log("PASS", "fetch_fmp_historical_price removed from data_fetcher")

# Verify URL pattern has apikey directly in string
if "&apikey={api_key}" in df_src or "&apikey={api_key}" in df_src:
    log("PASS", "fetch_fmp_dividends: apikey appended directly in URL string")
else:
    log("FAIL", "fetch_fmp_dividends: apikey not in URL string")

if "&apikey={api_key}" in tf_src:
    log("PASS", "test_fmp_connection: apikey appended directly in URL string")
else:
    log("FAIL", "test_fmp_connection: apikey not in URL string")

# Verify response parsing is flat list (not nested)
if '["historical"]' not in df_src and '["dividends"]' not in df_src:
    log("PASS", "fetch_fmp_dividends: no nested response access patterns")
else:
    log("FAIL", "fetch_fmp_dividends: still has nested response access")

if "isinstance(data, list)" in df_src:
    log("PASS", "fetch_fmp_dividends: checks for flat list response")
else:
    log("FAIL", "fetch_fmp_dividends: missing flat list check")


# ===========================================================================
# CHECK 3: Quarterly Rebalancing — verify Q4 present
# ===========================================================================
print("\n=== CHECK 3: Quarterly Rebalancing Dates ===")

from contributions import rebalance_dates

start_3yr = dt.date(2022, 1, 1)
end_3yr = dt.date(2024, 12, 31)
q_dates = rebalance_dates("Quarterly", start_3yr, end_3yr)

print(f"  Quarterly dates ({len(q_dates)} total):")
for d in q_dates:
    print(f"    {d}")

if len(q_dates) == 12:
    log("PASS", f"Quarterly: exactly 12 dates (4 per year x 3 years)")
else:
    log("FAIL", f"Quarterly: expected 12 dates, got {len(q_dates)}")

# Verify Q4 present in each year
for year in [2022, 2023, 2024]:
    q4_dates = [d for d in q_dates if d.year == year and d.month == 10]
    if q4_dates:
        log("PASS", f"Q4 {year}: found ({q4_dates[0]})")
    else:
        log("FAIL", f"Q4 {year}: MISSING")


# ===========================================================================
# CHECK 4: All Frequencies — 2-year window Jan 1 2023 – Dec 31 2024
# ===========================================================================
print("\n=== CHECK 4: All Rebalancing Frequencies (2-year window) ===")

start_2yr = dt.date(2023, 1, 1)
end_2yr = dt.date(2024, 12, 31)

expected = {
    "Daily": (450, 550),        # ~504 trading days
    "Weekly": (90, 115),        # ~104 dates
    "Bi-Weekly": (45, 60),      # ~52 dates
    "Semi-Monthly": (44, 52),   # ~48 dates
    "Monthly": (22, 26),        # 24 dates
    "Quarterly": (7, 9),        # 8 dates
    "Semi-Annually": (3, 5),    # 4 dates
    "Annually": (1, 3),         # 2 dates
}

for freq, (low, high) in expected.items():
    dates = rebalance_dates(freq, start_2yr, end_2yr)
    count = len(dates)
    if low <= count <= high:
        log("PASS", f"{freq}: {count} dates (expected {low}-{high})")
    else:
        log("FAIL", f"{freq}: {count} dates (expected {low}-{high})")
    # For quarterly, print the dates
    if freq == "Quarterly":
        for d in dates:
            print(f"    {d}")


# ===========================================================================
# CHECK 5: Sensitivity Removal
# ===========================================================================
print("\n=== CHECK 5: Yield Sensitivity Removal ===")

import subprocess
result = subprocess.run(
    ["grep", "-r", "sensitivity", ".", "--include=*.py", "-i", "-l"],
    capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
)
matches = [f for f in result.stdout.strip().split("\n") if f and "validate_fixes" not in f]
if not matches:
    log("PASS", "No 'sensitivity' references found in *.py files")
else:
    log("FAIL", f"Found 'sensitivity' in: {matches}")


# ===========================================================================
# CHECK 6: Empty Contributions Panel
# ===========================================================================
print("\n=== CHECK 6: Empty Contributions Default ===")

from portfolio import default_portfolio

pc = default_portfolio()
if len(pc.contributions) == 0:
    log("PASS", "Default portfolio has empty contributions list")
else:
    log("FAIL", f"Default portfolio has {len(pc.contributions)} contributions (expected 0)")


# ===========================================================================
# CHECK: DRIP price uses payment_date (structural)
# ===========================================================================
print("\n=== CHECK: DRIP Payment-Date Pricing (structural) ===")

import backtester
bt_src = inspect.getsource(backtester.run_backtest)
if "drip_price_date" in bt_src and "drip_price_source" in bt_src:
    log("PASS", "backtester DRIP logic tracks price_date and price_source")
else:
    log("FAIL", "backtester DRIP logic missing price_date/price_source tracking")

if "pay_ts = pd.Timestamp(pay_date)" in bt_src:
    log("PASS", "backtester looks up payment_date price from historical data")
else:
    log("FAIL", "backtester doesn't look up payment_date in price series")

# Check DividendRecord has drip_price_date field
from rebalancer import DividendRecord
dr = DividendRecord()
if hasattr(dr, "drip_price_date") and hasattr(dr, "drip_price_source"):
    log("PASS", "DividendRecord has drip_price_date and drip_price_source fields")
else:
    log("FAIL", "DividendRecord missing drip_price_date/drip_price_source fields")

# Check ui_contributions uses fetch_history for DRIP pricing
import ui_contributions
uc_src = inspect.getsource(ui_contributions.render_contributions)
if "fetch_history" in uc_src and "payment_date" in uc_src:
    log("PASS", "ui_contributions uses fetch_history for payment_date DRIP pricing")
else:
    log("FAIL", "ui_contributions doesn't use historical pricing for DRIP")

# Check diagnose_fmp_key exists in main
import main
if hasattr(main, "diagnose_fmp_key"):
    log("PASS", "diagnose_fmp_key function exists in main.py")
else:
    log("FAIL", "diagnose_fmp_key function missing from main.py")

# Cache bust check
main_src = inspect.getsource(main)
if "cache_data.clear()" in main_src and "fmp_diagnosed" in main_src:
    log("PASS", "main.py has one-time cache-bust via fmp_diagnosed flag")
else:
    log("FAIL", "main.py missing cache-bust logic")


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*60}")

sys.exit(1 if failed > 0 else 0)
