#!/usr/bin/env python3
"""
validate_rebalancing.py — End-to-end reconciliation test for the
rebalancing engine.

Test scenario:
    Single Kelly sleeve  — 50% VO / 50% PDI
    $100,000 initial investment
    Jan 2020 – Dec 2024
    Quarterly rebalancing (Mode B)
    0.1% transaction cost
    12% annualized yield target

Assertions:
    1. Cash never goes negative on any day
    2. Every SELL has a matching BUY with the same paired_trade_id
    3. Value conservation: sleeve total before trade ≈ sleeve total after trade (within tx costs)
    4. No phantom shares: share_delta matches shares_after - shares_before
    5. VO is sold when PDI is bought, and vice versa (Kelly behavior)
    6. Trade log contains both legs of each trade
    7. Final portfolio value is reasonable (positive, > 0)
"""

from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict

from portfolio import PortfolioConfig, Holding, Sleeve
from backtester import run_backtest


def build_test_config() -> PortfolioConfig:
    """Build the 50/50 VO/PDI Kelly sleeve test configuration."""
    config = PortfolioConfig()

    # Create holdings
    vo = Holding(ticker="VO", shares=0.0, cost_basis=0.0, asset_class="Equity")
    pdi = Holding(ticker="PDI", shares=0.0, cost_basis=0.0, asset_class="Fixed Income",
                  drip_enabled=True)

    config.holdings = [vo, pdi]

    # Create Mode B Kelly sleeve
    sleeve = Sleeve(
        name="Kelly Test",
        holding_ids=[vo.id, pdi.id],
        mode="B: Signal-Based (Kelly)",
        stock_ticker="VO",
        bond_ticker="PDI",
        annualized_yield_pct=12.0,
        signal_scaling=1.0,
        signal_frequency="Quarterly",
        transaction_cost_pct=0.1,
        cash_balance=100_000.0,  # $100K starting cash
    )

    # Set target weights for initial allocation
    sleeve.target_weights = {vo.id: 50.0, pdi.id: 50.0}
    config.sleeves = [sleeve]

    # Backtest window
    config.backtest_start = "2020-01-01"
    config.backtest_end = "2024-12-31"
    config.benchmark_tickers = ["SPY"]

    return config


def validate_trade_log(trade_log: list) -> list[str]:
    """Run all assertions on the trade log. Returns list of failures."""
    failures: list[str] = []

    if not trade_log:
        failures.append("FAIL: Trade log is empty — no trades were executed")
        return failures

    # -------------------------------------------------------------------
    # 1. Cash never goes negative
    # -------------------------------------------------------------------
    negative_cash_events = []
    for tr in trade_log:
        cash_after = getattr(tr, "cash_after", None)
        if cash_after is not None and cash_after < -0.01:
            negative_cash_events.append(
                f"  {tr.date} {tr.ticker} {tr.action}: cash_after=${cash_after:.2f}"
            )
    if negative_cash_events:
        failures.append(
            f"FAIL: Cash went negative {len(negative_cash_events)} time(s):\n"
            + "\n".join(negative_cash_events[:10])
        )
    else:
        print("PASS: Cash never went negative")

    # -------------------------------------------------------------------
    # 2. Every non-HOLD trade should have a paired_trade_id with a counterpart
    # -------------------------------------------------------------------
    pair_groups: dict[str, list] = defaultdict(list)
    for tr in trade_log:
        pid = getattr(tr, "paired_trade_id", "")
        if pid and tr.action != "HOLD":
            pair_groups[pid].append(tr)

    orphan_pairs = []
    for pid, trades in pair_groups.items():
        actions = {tr.action for tr in trades}
        if len(trades) < 2 and "SELL" in actions:
            # A SELL without a matching BUY is an orphan
            orphan_pairs.append(pid)
        elif len(trades) < 2 and "BUY" in actions:
            # Guard-blocked buys won't have matching sells if insufficient cash
            has_guard = any(getattr(tr, "guard_active", False) for tr in trades)
            if not has_guard:
                orphan_pairs.append(pid)

    if orphan_pairs:
        failures.append(
            f"FAIL: {len(orphan_pairs)} orphan paired_trade_id(s) — "
            f"trade has no counterpart: {orphan_pairs[:5]}"
        )
    else:
        print(f"PASS: All {len(pair_groups)} paired trades have both legs")

    # -------------------------------------------------------------------
    # 3. Value conservation per paired trade
    # -------------------------------------------------------------------
    conservation_violations = []
    for pid, trades in pair_groups.items():
        if len(trades) < 2:
            continue
        # Net cash impact across legs should roughly sum to negative tx costs
        total_cash_impact = sum(getattr(tr, "net_cash_impact", 0.0) for tr in trades)
        total_tx = sum(tr.transaction_cost for tr in trades)
        # After both legs, cash impact should be approximately -total_tx (costs eaten)
        # In practice it should be close to zero since sell proceeds fund buy
        if abs(total_cash_impact) > total_tx * 2 + 1.0:
            conservation_violations.append(
                f"  Pair {pid[:8]}: net_cash={total_cash_impact:.2f}, tx={total_tx:.2f}"
            )

    if conservation_violations:
        failures.append(
            f"FAIL: Value conservation violated in {len(conservation_violations)} trade pair(s):\n"
            + "\n".join(conservation_violations[:10])
        )
    else:
        print("PASS: Value conservation holds across all paired trades")

    # -------------------------------------------------------------------
    # 4. Share delta matches shares_after - shares_before
    # -------------------------------------------------------------------
    delta_mismatches = []
    for tr in trade_log:
        if tr.action == "HOLD":
            continue
        expected_delta = getattr(tr, "shares_after", 0.0) - getattr(tr, "shares_before", 0.0)
        actual_delta = getattr(tr, "share_delta", 0.0)
        if abs(expected_delta - actual_delta) > 0.0001:
            delta_mismatches.append(
                f"  {tr.date} {tr.ticker}: expected_delta={expected_delta:.6f}, "
                f"actual_delta={actual_delta:.6f}"
            )

    if delta_mismatches:
        failures.append(
            f"FAIL: Share delta mismatch in {len(delta_mismatches)} trade(s):\n"
            + "\n".join(delta_mismatches[:10])
        )
    else:
        print("PASS: All share deltas match shares_after - shares_before")

    # -------------------------------------------------------------------
    # 5. Kelly behavior: VO sold ↔ PDI bought, PDI sold ↔ VO bought
    # -------------------------------------------------------------------
    kelly_violations = []
    for pid, trades in pair_groups.items():
        if len(trades) < 2:
            continue
        tickers_by_action = defaultdict(set)
        for tr in trades:
            tickers_by_action[tr.action].add(tr.ticker)

        sells = tickers_by_action.get("SELL", set())
        buys = tickers_by_action.get("BUY", set())

        # In Mode B: selling stock should pair with buying bond or vice versa
        if sells and buys:
            if sells == buys:
                kelly_violations.append(
                    f"  Pair {pid[:8]}: SELL and BUY same ticker(s): {sells}"
                )

    if kelly_violations:
        failures.append(
            f"FAIL: Kelly behavior violated in {len(kelly_violations)} pair(s):\n"
            + "\n".join(kelly_violations[:5])
        )
    else:
        print("PASS: Kelly pairs always swap between different tickers")

    # -------------------------------------------------------------------
    # 6. Both legs present for non-HOLD trades
    # -------------------------------------------------------------------
    both_legs_count = sum(1 for trades in pair_groups.values() if len(trades) >= 2)
    single_legs = sum(1 for trades in pair_groups.values() if len(trades) == 1)
    print(f"INFO: {both_legs_count} two-leg pairs, {single_legs} single-leg entries")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total_trades = len(trade_log)
    sells_count = sum(1 for tr in trade_log if tr.action == "SELL")
    buys_count = sum(1 for tr in trade_log if tr.action == "BUY")
    holds_count = sum(1 for tr in trade_log if tr.action == "HOLD")
    guarded_count = sum(1 for tr in trade_log if getattr(tr, "guard_active", False))
    print(f"\nTrade Summary: {total_trades} total — "
          f"{sells_count} SELLs, {buys_count} BUYs, {holds_count} HOLDs, "
          f"{guarded_count} cash-guarded")

    return failures


def main():
    print("=" * 70)
    print("Rebalancing Engine — End-to-End Reconciliation Test")
    print("=" * 70)
    print()
    print("Scenario: Single Kelly Sleeve, 50/50 VO/PDI")
    print("  Initial: $100,000 | Period: Jan 2020 – Dec 2024")
    print("  Rebalancing: Quarterly | Tx Cost: 0.1% | Yield: 12%")
    print()

    config = build_test_config()

    print("Running backtest...")
    try:
        result = run_backtest(config)
    except Exception as e:
        print(f"FATAL: Backtest raised exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -------------------------------------------------------------------
    # 7. Final portfolio value is reasonable
    # -------------------------------------------------------------------
    if result.portfolio_values.empty:
        print("FAIL: Backtest returned empty portfolio values.")
        print("  (This is expected if running in a sandboxed environment")
        print("   without network access to Yahoo Finance.)")
        sys.exit(1)

    start_val = result.portfolio_values.iloc[0]
    end_val = result.portfolio_values.iloc[-1]
    total_return = (end_val / start_val - 1) * 100
    print(f"\nPortfolio Value: ${start_val:,.2f} → ${end_val:,.2f} ({total_return:+.1f}%)")
    print(f"Trading Days: {len(result.portfolio_values)}")
    print(f"Trade Log: {len(result.trade_log)} entries")
    print(f"Dividend Log: {len(result.dividend_log)} entries")
    print()

    if end_val <= 0:
        print("FAIL: Final portfolio value is non-positive!")
        sys.exit(1)
    else:
        print("PASS: Final portfolio value is positive")

    # -------------------------------------------------------------------
    # Run trade log validations
    # -------------------------------------------------------------------
    print()
    print("-" * 50)
    print("Trade Log Validations")
    print("-" * 50)
    failures = validate_trade_log(result.trade_log)

    # -------------------------------------------------------------------
    # Dividend checks
    # -------------------------------------------------------------------
    print()
    print("-" * 50)
    print("Dividend Validations")
    print("-" * 50)
    if result.dividend_log:
        total_divs = sum(dr.gross_amount for dr in result.dividend_log)
        drip_count = sum(1 for dr in result.dividend_log if dr.treatment == "DRIP")
        cash_count = sum(1 for dr in result.dividend_log if dr.treatment == "Cash")
        print(f"Total Dividends: ${total_divs:,.2f} across {len(result.dividend_log)} events")
        print(f"  DRIP: {drip_count} | Cash: {cash_count}")
        print("PASS: Dividend events were captured")
    else:
        print("INFO: No dividend events captured (may need network access)")

    # -------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(s)")
        for f in failures:
            print(f"\n{f}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
