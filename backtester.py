"""
backtester.py — Historical backtesting engine with multi-sleeve support
and full dividend / distribution handling (DRIP + Cash accrual).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from contributions import (
    build_contribution_schedule,
    build_withdrawal_schedule,
    rebalance_dates,
    frequency_to_annual_fraction,
)
from data_fetcher import (
    fetch_multiple_histories,
    fetch_dividend_history,
    fetch_multiple_dividend_histories,
    get_dividend_events,
    compute_standardized_yield,
)
from portfolio import PortfolioConfig, Sleeve
from rebalancer import (
    DividendRecord,
    SleeveState,
    TradeRecord,
    apply_rebalancing,
)


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    portfolio_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_values: dict[str, pd.Series] = field(default_factory=dict)
    trade_log: list[TradeRecord] = field(default_factory=list)
    dividend_log: list[DividendRecord] = field(default_factory=list)
    sleeve_values: dict[str, pd.Series] = field(default_factory=dict)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Dividend event pre-processing helpers
# ---------------------------------------------------------------------------

def _build_dividend_schedule(
    tickers: list[str],
    config: PortfolioConfig,
    start_str: str,
    end_str: str,
) -> dict[str, pd.DataFrame]:
    """Build a per-ticker dividend events DataFrame for the backtest window.

    Returns ``{ticker: DataFrame}`` where each DataFrame has:
        ex_dividend_date, payment_date, payment_date_source,
        declaration_date, record_date, dividend_per_share
    """
    schedules: dict[str, pd.DataFrame] = {}
    for t in tickers:
        h = config.holding_by_ticker(t)
        offset = h.payment_date_offset if h else 20
        events = get_dividend_events(t, start=start_str, end=end_str,
                                     payment_date_offset=offset)
        if not events.empty:
            schedules[t] = events
    return schedules


def _build_ex_date_lookup(
    schedules: dict[str, pd.DataFrame],
) -> dict[dt.date, list[dict]]:
    """Index dividend events by ex-dividend date for fast daily lookup.

    Returns ``{date: [event_dict, ...]}`` where each dict has all dividend
    event columns plus ``ticker``.
    """
    lookup: dict[dt.date, list[dict]] = {}
    for ticker, df in schedules.items():
        for _, row in df.iterrows():
            ex_d = row["ex_dividend_date"]
            if not isinstance(ex_d, dt.date):
                ex_d = pd.Timestamp(ex_d).date()
            entry = row.to_dict()
            entry["ticker"] = ticker
            lookup.setdefault(ex_d, []).append(entry)
    return lookup


# ---------------------------------------------------------------------------
# Core backtesting engine
# ---------------------------------------------------------------------------

def run_backtest(config: PortfolioConfig) -> BacktestResult:
    """Run a historical backtest with full dividend handling.

    Dividend recognition follows the ex-dividend date; DRIP share
    purchases execute on the payment date.
    """
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
        tickers = [t for t in tickers if t in prices_df.columns]
        if not tickers:
            return result

    prices_df = prices_df.ffill().dropna(how="all")
    dates = prices_df.index

    # ------------------------------------------------------------------
    # Fetch dividend schedules for all holding tickers
    # ------------------------------------------------------------------
    div_schedules = _build_dividend_schedule(tickers, config, start_str, end_str)
    ex_date_lookup = _build_ex_date_lookup(div_schedules)

    # Build DRIP-enabled lookup from config
    drip_settings: dict[str, bool] = {}
    for h in config.holdings:
        if h.ticker:
            drip_settings[h.ticker] = h.drip_enabled

    # Pre-compute real yield data for Mode C formulas
    yield_data: dict[str, float] = {}
    for t in tickers:
        y, _, _ = compute_standardized_yield(t)
        yield_data[t] = y

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
    unmanaged_div_cash: float = 0.0
    for h in config.unassigned_holdings():
        if h.ticker and h.ticker in prices_df.columns:
            unmanaged_positions[h.ticker] = h.shares

    # Pre-compute rebalance date sets per sleeve
    sleeve_rebal_dates: dict[str, set] = {}
    for sleeve in config.sleeves:
        freq = sleeve.rebalance_trigger if sleeve.mode.startswith("A") else sleeve.signal_frequency
        if freq in ("Threshold",):
            sleeve_rebal_dates[sleeve.id] = set(
                d.date() if hasattr(d, 'date') else d for d in dates
            )
        else:
            rdates = rebalance_dates(freq, start_date, end_date)
            sleeve_rebal_dates[sleeve.id] = set(rdates)

    # ------------------------------------------------------------------
    # Run day-by-day simulation
    # ------------------------------------------------------------------
    portfolio_vals = []
    sleeve_val_series: dict[str, list] = {s.id: [] for s in config.sleeves}
    trade_log: list[TradeRecord] = []
    dividend_log: list[DividendRecord] = []

    for i, ts in enumerate(dates):
        d = ts.date() if hasattr(ts, 'date') else ts
        prices_today = {
            t: float(prices_df[t].iloc[i])
            for t in prices_df.columns
            if not pd.isna(prices_df[t].iloc[i])
        }

        # ==============================================================
        # 1) Process DRIP purchases scheduled for today (payment date)
        # ==============================================================
        for sid, ss in sleeve_states.items():
            tickers_to_process = list(ss.pending_drip.keys())
            for ticker in tickers_to_process:
                pending = ss.pending_drip[ticker]
                still_pending = []
                for pay_date, amount, ex_date_ref in pending:
                    if d >= pay_date:
                        price = prices_today.get(ticker, 0.0)
                        if price > 0:
                            new_shares = amount / price
                            ss.positions[ticker] = ss.positions.get(ticker, 0.0) + new_shares
                            ss.cash -= amount
                            # Update the corresponding DividendRecord
                            for dr in dividend_log:
                                if (dr.ticker == ticker
                                        and dr.ex_dividend_date == ex_date_ref
                                        and dr.sleeve_name == ss.sleeve.name
                                        and dr.drip_shares == 0.0):
                                    dr.drip_shares = new_shares
                                    dr.drip_price = price
                                    break
                    else:
                        still_pending.append((pay_date, amount, ex_date_ref))
                ss.pending_drip[ticker] = still_pending

        # ==============================================================
        # 2) Process ex-dividend events for today
        # ==============================================================
        if d in ex_date_lookup:
            for event in ex_date_lookup[d]:
                ticker = event["ticker"]
                div_ps = float(event["dividend_per_share"])
                pay_d = event.get("payment_date", d + dt.timedelta(days=20))
                if pay_d is None:
                    pay_d = d + dt.timedelta(days=20)
                if not isinstance(pay_d, dt.date):
                    pay_d = pd.Timestamp(pay_d).date()
                pay_src = event.get("payment_date_source", "Estimated")
                decl_d = event.get("declaration_date")
                rec_d = event.get("record_date")

                is_drip = drip_settings.get(ticker, True)

                # Process across all sleeves that hold this ticker
                for sid, ss in sleeve_states.items():
                    shares = ss.positions.get(ticker, 0.0)
                    if shares <= 0:
                        continue
                    gross = div_ps * shares
                    # Add dividend to cash on ex-date (return recognition)
                    ss.cash += gross

                    dr = DividendRecord(
                        ex_dividend_date=d,
                        payment_date=pay_d,
                        payment_date_source=pay_src,
                        declaration_date=decl_d if isinstance(decl_d, dt.date) else None,
                        record_date=rec_d if isinstance(rec_d, dt.date) else None,
                        sleeve_name=ss.sleeve.name,
                        ticker=ticker,
                        dividend_per_share=div_ps,
                        shares_held=shares,
                        gross_amount=gross,
                        treatment="DRIP" if is_drip else "Cash",
                    )

                    if is_drip:
                        # Schedule DRIP purchase on payment date
                        ss.pending_drip.setdefault(ticker, []).append(
                            (pay_d, gross, d)
                        )
                    else:
                        dr.cash_added = gross

                    dividend_log.append(dr)

                # Unmanaged holdings — treat as cash
                unmanaged_shares = unmanaged_positions.get(ticker, 0.0)
                if unmanaged_shares > 0:
                    gross = div_ps * unmanaged_shares
                    unmanaged_div_cash += gross
                    dividend_log.append(DividendRecord(
                        ex_dividend_date=d,
                        payment_date=pay_d,
                        payment_date_source=pay_src,
                        declaration_date=decl_d if isinstance(decl_d, dt.date) else None,
                        record_date=rec_d if isinstance(rec_d, dt.date) else None,
                        sleeve_name="Unmanaged",
                        ticker=ticker,
                        dividend_per_share=div_ps,
                        shares_held=unmanaged_shares,
                        gross_amount=gross,
                        treatment="Cash",
                        cash_added=gross,
                    ))

        # ==============================================================
        # 3) Apply contributions for today
        # ==============================================================
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

            # ==============================================================
            # 4) Check if today is a rebalance date for this sleeve
            # ==============================================================
            if d in sleeve_rebal_dates.get(sid, set()):
                trades = apply_rebalancing(
                    ss, prices_today, d, config, sleeve_cf,
                    yield_data=yield_data,
                )
                trade_log.extend(trades)
            else:
                ss.cash += sleeve_cf

        # ==============================================================
        # 5) Compute total portfolio value
        # ==============================================================
        total_val = 0.0
        for sid, ss in sleeve_states.items():
            sv = ss.total_value(prices_today)
            sleeve_val_series[sid].append(sv)
            total_val += sv

        # Add unmanaged positions value + accumulated unmanaged dividend cash
        for t, shares in unmanaged_positions.items():
            total_val += shares * prices_today.get(t, 0.0)
        total_val += unmanaged_div_cash

        portfolio_vals.append(total_val)

    # ------------------------------------------------------------------
    # Build result series
    # ------------------------------------------------------------------
    result.portfolio_values = pd.Series(
        portfolio_vals, index=dates, name="Portfolio",
    )
    result.daily_returns = result.portfolio_values.pct_change().dropna()
    result.trade_log = trade_log
    result.dividend_log = dividend_log

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
