"""
backtester.py — Historical backtesting engine with multi-sleeve support.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from contributions import (
    build_contribution_schedule,
    build_withdrawal_schedule,
    rebalance_dates,
    frequency_to_annual_fraction,
)
from data_fetcher import fetch_multiple_histories
from portfolio import PortfolioConfig, Sleeve
from rebalancer import (
    SleeveState,
    TradeRecord,
    apply_rebalancing,
    rebalance_mode_b,
)


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    portfolio_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_values: dict[str, pd.Series] = field(default_factory=dict)
    trade_log: list[TradeRecord] = field(default_factory=list)
    sleeve_values: dict[str, pd.Series] = field(default_factory=dict)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Core backtesting engine
# ---------------------------------------------------------------------------

def run_backtest(config: PortfolioConfig) -> BacktestResult:
    """Run a historical backtest. Returns BacktestResult."""
    result = BacktestResult()

    start_str = config.backtest_start
    end_str = config.backtest_end
    if not start_str or not end_str:
        return result

    start_date = dt.date.fromisoformat(start_str)
    end_date = dt.date.fromisoformat(end_str)
    if start_date >= end_date:
        return result

    # Collect all tickers
    tickers = config.tickers()
    if not tickers:
        return result

    # Add benchmark tickers
    all_tickers = list(set(tickers + config.benchmark_tickers))
    prices_df = fetch_multiple_histories(all_tickers, start=start_str, end=end_str)
    if prices_df.empty:
        return result

    # Ensure all holding tickers are present
    missing = [t for t in tickers if t not in prices_df.columns]
    if missing:
        # Drop missing tickers from consideration
        tickers = [t for t in tickers if t in prices_df.columns]
        if not tickers:
            return result

    prices_df = prices_df.ffill().dropna(how="all")
    dates = prices_df.index

    # Build contribution/withdrawal schedules
    contrib_sched = build_contribution_schedule(
        config.contributions, start_date, end_date,
    )
    withdrawal_sched = build_withdrawal_schedule(
        config.withdrawals, start_date, end_date,
    )

    # Initialize sleeve states
    sleeve_states: dict[str, SleeveState] = {}
    for sleeve in config.sleeves:
        ss = SleeveState(sleeve=sleeve, cash=sleeve.cash_balance)
        for hid in sleeve.holding_ids:
            h = config.holding_by_id(hid)
            if h and h.ticker and h.ticker in prices_df.columns:
                first_price = prices_df[h.ticker].iloc[0]
                if first_price > 0:
                    ss.positions[h.ticker] = h.shares
        sleeve_states[sleeve.id] = ss

    # Unmanaged holdings
    unmanaged_positions: dict[str, float] = {}
    for h in config.unassigned_holdings():
        if h.ticker and h.ticker in prices_df.columns:
            unmanaged_positions[h.ticker] = h.shares

    # Pre-compute rebalance date sets per sleeve
    sleeve_rebal_dates: dict[str, set] = {}
    for sleeve in config.sleeves:
        freq = sleeve.rebalance_trigger if sleeve.mode.startswith("A") else sleeve.signal_frequency
        if freq in ("Threshold",):
            # Threshold-only: check every day
            sleeve_rebal_dates[sleeve.id] = set(d.date() if hasattr(d, 'date') else d for d in dates)
        else:
            rdates = rebalance_dates(freq, start_date, end_date)
            sleeve_rebal_dates[sleeve.id] = set(rdates)

    # Run day-by-day simulation
    portfolio_vals = []
    sleeve_val_series: dict[str, list] = {s.id: [] for s in config.sleeves}
    trade_log: list[TradeRecord] = []

    for i, ts in enumerate(dates):
        d = ts.date() if hasattr(ts, 'date') else ts
        prices_today = {t: float(prices_df[t].iloc[i])
                        for t in prices_df.columns
                        if not pd.isna(prices_df[t].iloc[i])}

        # Apply contributions for today
        contrib_today = 0.0
        if not contrib_sched.empty:
            mask = contrib_sched["date"] == d
            contrib_today = float(contrib_sched.loc[mask, "amount"].sum())

        # Apply withdrawals for today
        withdrawal_today = 0.0
        if not withdrawal_sched.empty:
            mask = withdrawal_sched["date"] == d
            for _, row in withdrawal_sched[mask].iterrows():
                if row["is_percentage"]:
                    # Estimate current portfolio value
                    est_val = sum(
                        ss.total_value(prices_today)
                        for ss in sleeve_states.values()
                    ) + sum(
                        unmanaged_positions.get(t, 0) * prices_today.get(t, 0)
                        for t in unmanaged_positions
                    )
                    withdrawal_today += est_val * (row["amount"] / 100.0)
                else:
                    withdrawal_today += row["amount"]

        net_cf = contrib_today - withdrawal_today

        # Distribute cashflow across sleeves proportionally
        total_sleeve_val = sum(ss.total_value(prices_today) for ss in sleeve_states.values())
        for sid, ss in sleeve_states.items():
            sv = ss.total_value(prices_today)
            if total_sleeve_val > 0:
                sleeve_cf = net_cf * (sv / total_sleeve_val)
            elif len(sleeve_states) > 0:
                sleeve_cf = net_cf / len(sleeve_states)
            else:
                sleeve_cf = 0.0

            # Check if today is a rebalance date for this sleeve
            if d in sleeve_rebal_dates.get(sid, set()):
                trades = apply_rebalancing(
                    ss, prices_today, d, config, sleeve_cf,
                )
                trade_log.extend(trades)
            else:
                ss.cash += sleeve_cf

        # Compute total portfolio value
        total_val = 0.0
        for sid, ss in sleeve_states.items():
            sv = ss.total_value(prices_today)
            sleeve_val_series[sid].append(sv)
            total_val += sv

        # Add unmanaged positions value
        for t, shares in unmanaged_positions.items():
            total_val += shares * prices_today.get(t, 0.0)

        portfolio_vals.append(total_val)

    # Build result series
    result.portfolio_values = pd.Series(
        portfolio_vals, index=dates, name="Portfolio",
    )
    result.daily_returns = result.portfolio_values.pct_change().dropna()
    result.trade_log = trade_log

    for sid, vals in sleeve_val_series.items():
        sleeve = next((s for s in config.sleeves if s.id == sid), None)
        name = sleeve.name if sleeve else sid
        result.sleeve_values[name] = pd.Series(vals, index=dates, name=name)

    # Benchmark values (normalized to same starting value)
    start_val = portfolio_vals[0] if portfolio_vals else 1.0
    for bt in config.benchmark_tickers:
        if bt in prices_df.columns:
            bench = prices_df[bt].copy()
            bench = bench / bench.iloc[0] * start_val
            result.benchmark_values[bt] = bench

    return result


# ---------------------------------------------------------------------------
# Mode B yield sensitivity analysis
# ---------------------------------------------------------------------------

def run_yield_sensitivity(
    config: PortfolioConfig,
    sleeve: Sleeve,
    yield_range: list[float] = None,
) -> pd.DataFrame:
    """Re-run backtest across a range of yield assumptions for Mode B sleeve.
    Returns DataFrame with columns: Yield, CAGR, Volatility, Max Drawdown, Sharpe."""
    from analytics import (
        annualized_cagr,
        annualized_volatility,
        max_drawdown,
        sharpe_ratio,
    )

    if yield_range is None:
        yield_range = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    rows = []
    original_yield = sleeve.annualized_yield_pct

    for y in yield_range:
        sleeve.annualized_yield_pct = y
        result = run_backtest(config)
        if result.portfolio_values.empty:
            continue
        rets = result.portfolio_values.pct_change().dropna()
        dd, _, _ = max_drawdown(result.portfolio_values)
        rows.append({
            "Yield (%)": y,
            "CAGR (%)": annualized_cagr(result.portfolio_values) * 100,
            "Volatility (%)": annualized_volatility(rets) * 100,
            "Max Drawdown (%)": dd * 100,
            "Sharpe Ratio": sharpe_ratio(rets),
        })

    sleeve.annualized_yield_pct = original_yield

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
