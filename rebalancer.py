"""
rebalancer.py — Multi-sleeve rebalancing engine (Modes A, B, C)
with double-entry cash ledger, paired trade tracking, and dividend
event handling (DRIP / Cash accrual).

Every trade now routes through the sleeve's cash balance:
  SELL → proceeds credited to cash
  BUY  → funded from cash (never goes negative)

Each trade logs both sides with a shared ``paired_trade_id`` and
carries full before/after snapshots for reconciliation.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from asteval import Interpreter

from contributions import frequency_to_annual_fraction, rebalance_dates
from portfolio import Sleeve, PortfolioConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enhanced Trade Record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Full audit-trail record for a single leg of a rebalancing event."""
    date: dt.date = dt.date.today()
    sleeve_name: str = ""
    event_type: str = ""       # "Rebalance-A" / "Rebalance-B" / "Rebalance-C" / "Contribution" / "DRIP"
    ticker: str = ""
    action: str = ""           # "BUY" / "SELL" / "HOLD"
    shares_before: float = 0.0
    shares_after: float = 0.0
    share_delta: float = 0.0
    execution_price: float = 0.0
    gross_trade_value: float = 0.0
    transaction_cost: float = 0.0
    net_cash_impact: float = 0.0   # positive = cash inflow, negative = cash outflow
    cash_before: float = 0.0
    cash_after: float = 0.0
    funding_source: str = ""       # "SELL proceeds" / "Sleeve cash" / "Contribution" / "Dividend"
    paired_trade_id: str = ""      # shared ID linking SELL leg to BUY leg
    signal_amount: float = 0.0
    guard_active: bool = False     # True if cash-guard prevented full execution
    shortfall_amount: float = 0.0  # how much more cash was needed
    note: str = ""

    # Backward-compat properties for old UI code
    @property
    def direction(self) -> str:
        return self.action

    @property
    def shares(self) -> float:
        return abs(self.share_delta)

    @property
    def dollar_amount(self) -> float:
        return self.gross_trade_value


# ---------------------------------------------------------------------------
# Dividend record (separate from trades for clarity)
# ---------------------------------------------------------------------------

@dataclass
class DividendRecord:
    """Tracks a single dividend / distribution event."""
    ex_dividend_date: dt.date = dt.date.today()
    payment_date: dt.date = dt.date.today()
    payment_date_source: str = "Estimated"  # "FMP" / "Estimated"
    declaration_date: Optional[dt.date] = None
    record_date: Optional[dt.date] = None
    sleeve_name: str = ""
    ticker: str = ""
    dividend_per_share: float = 0.0
    shares_held: float = 0.0
    gross_amount: float = 0.0
    treatment: str = "DRIP"  # "DRIP" / "Cash"
    drip_shares: float = 0.0
    drip_price: float = 0.0
    drip_price_date: Optional[dt.date] = None
    drip_price_source: str = ""  # "Historical" / "Estimated"
    cash_added: float = 0.0


# ---------------------------------------------------------------------------
# Sleeve state during simulation
# ---------------------------------------------------------------------------

@dataclass
class SleeveState:
    sleeve: Sleeve = field(default_factory=Sleeve)
    positions: dict[str, float] = field(default_factory=dict)  # ticker → shares
    cash: float = 0.0
    prev_stock_value: float = 0.0  # for Mode B tracking
    # Pending DRIP amounts: ticker → [(payment_date, dollar_amount, ex_date_ref), ...]
    pending_drip: dict[str, list] = field(default_factory=dict)

    def position_value(self, prices: dict[str, float]) -> dict[str, float]:
        """Dollar value per ticker."""
        return {t: self.positions.get(t, 0.0) * prices.get(t, 0.0)
                for t in self.positions}

    def total_value(self, prices: dict[str, float]) -> float:
        return sum(self.position_value(prices).values()) + self.cash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_pair_id() -> str:
    """Generate a unique paired-trade identifier."""
    return uuid.uuid4().hex[:12]


def _assert_conservation(
    label: str,
    value_before: float,
    value_after: float,
    tolerance: float = 0.02,
) -> None:
    """Warn if total sleeve value changed beyond transaction costs + tolerance."""
    diff = abs(value_after - value_before)
    # Allow up to 2% drift for tx costs and rounding
    if value_before > 0 and diff / value_before > tolerance:
        logger.warning(
            "Value conservation check (%s): before=%.2f after=%.2f diff=%.2f (%.2f%%)",
            label, value_before, value_after, diff, diff / value_before * 100,
        )


def _hid_to_ticker_map(sleeve: Sleeve, config: PortfolioConfig) -> dict[str, str]:
    """Build holding_id → ticker mapping for a sleeve."""
    mapping: dict[str, str] = {}
    for hid in sleeve.holding_ids:
        h = config.holding_by_id(hid)
        if h and h.ticker:
            mapping[hid] = h.ticker
    return mapping


# ---------------------------------------------------------------------------
# Mode A: Target-Weight Rebalancing
# ---------------------------------------------------------------------------

def rebalance_mode_a(
    state: SleeveState,
    prices: dict[str, float],
    date: dt.date,
    config: PortfolioConfig,
    trigger_type: str,
    threshold_pct: float,
    method: str,
    tx_cost_pct: float,
    contribution_cash: float = 0.0,
) -> list[TradeRecord]:
    """Perform Mode A rebalancing with proper sell-before-buy ordering.

    Key fixes over the previous version:
    1. Translates holding IDs in target_weights to tickers
    2. Executes SELLs first to generate cash, then BUYs
    3. Cash can never go negative — shortfalls are logged
    """
    trades: list[TradeRecord] = []
    sleeve = state.sleeve

    # Add contribution cash
    state.cash += contribution_cash

    total = state.total_value(prices)
    if total <= 0:
        return trades

    # Build hid → ticker map; target_weights keys are holding IDs
    hid_to_ticker = _hid_to_ticker_map(sleeve, config)

    # Translate target_weights from holding_id → ticker
    ticker_targets: dict[str, float] = {}
    for hid, weight in sleeve.target_weights.items():
        ticker = hid_to_ticker.get(hid)
        if ticker:
            ticker_targets[ticker] = weight

    if not ticker_targets:
        return trades

    # Current weights within the sleeve
    pos_vals = state.position_value(prices)
    current_weights = {t: v / total if total > 0 else 0.0 for t, v in pos_vals.items()}

    # Check threshold trigger
    if trigger_type in ("Threshold", "Hybrid"):
        max_drift = max(
            abs(current_weights.get(t, 0.0) - ticker_targets.get(t, 0.0) / 100.0)
            for t in ticker_targets
        )
        if max_drift * 100 < threshold_pct:
            return trades

    # Compute target values
    target_vals = {t: total * (w / 100.0) for t, w in ticker_targets.items()}

    # --- Contributions-only mode ---
    if method == "Contributions-only":
        available = state.cash
        if available <= 0:
            return trades
        underweight = {
            t: max(0, target_vals.get(t, 0.0) - pos_vals.get(t, 0.0))
            for t in ticker_targets
        }
        total_uw = sum(underweight.values())
        if total_uw <= 0:
            return trades
        pair_id = _new_pair_id()
        for t, uw in underweight.items():
            if uw <= 0:
                continue
            alloc = min(available, available * (uw / total_uw))
            cost = alloc * (tx_cost_pct / 100.0)
            net = alloc - cost
            price = prices.get(t, 0.0)
            if net <= 0 or price <= 0:
                continue
            shares_buy = net / price
            cash_before = state.cash
            shares_before = state.positions.get(t, 0.0)
            state.positions[t] = shares_before + shares_buy
            state.cash -= alloc
            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-A",
                ticker=t, action="BUY",
                shares_before=shares_before, shares_after=state.positions[t],
                share_delta=shares_buy, execution_price=price,
                gross_trade_value=net, transaction_cost=cost,
                net_cash_impact=-alloc, cash_before=cash_before, cash_after=state.cash,
                funding_source="Sleeve cash", paired_trade_id=pair_id,
                note="Contributions-only rebalance",
            ))
        return trades

    # --- Full or Partial rebalancing ---
    # Separate into sells and buys
    sells: list[tuple[str, float]] = []  # (ticker, diff_amount)  diff < 0 means sell
    buys: list[tuple[str, float]] = []   # (ticker, diff_amount)  diff > 0 means buy

    for t, tw in ticker_targets.items():
        target_val = target_vals[t]
        current_val = pos_vals.get(t, 0.0)
        diff = target_val - current_val

        if method == "Partial" and abs(diff) / total * 100 < threshold_pct:
            continue
        if abs(diff) < 1.0:
            continue
        price = prices.get(t, 0.0)
        if price <= 0:
            continue

        if diff < 0:
            sells.append((t, diff))
        else:
            buys.append((t, diff))

    value_before = state.total_value(prices)
    pair_id = _new_pair_id()

    # Execute SELLs first to generate cash
    for t, diff in sells:
        price = prices.get(t, 0.0)
        if price <= 0:
            continue
        sell_value = abs(diff)
        shares_sell = min(sell_value / price, state.positions.get(t, 0.0))
        if shares_sell <= 0:
            continue
        gross = shares_sell * price
        cost = gross * (tx_cost_pct / 100.0)
        proceeds = gross - cost

        cash_before = state.cash
        shares_before = state.positions.get(t, 0.0)
        state.positions[t] = shares_before - shares_sell
        state.cash += proceeds

        trades.append(TradeRecord(
            date=date, sleeve_name=sleeve.name, event_type="Rebalance-A",
            ticker=t, action="SELL",
            shares_before=shares_before, shares_after=state.positions[t],
            share_delta=-shares_sell, execution_price=price,
            gross_trade_value=gross, transaction_cost=cost,
            net_cash_impact=proceeds, cash_before=cash_before, cash_after=state.cash,
            funding_source="Position liquidation", paired_trade_id=pair_id,
        ))

    # Execute BUYs from available cash
    for t, diff in buys:
        price = prices.get(t, 0.0)
        if price <= 0:
            continue

        desired_buy = diff  # dollar amount we want to buy
        cost = desired_buy * (tx_cost_pct / 100.0)
        total_needed = desired_buy + cost

        # Cash guard: cap to available cash
        guard_active = False
        shortfall = 0.0
        if total_needed > state.cash:
            shortfall = total_needed - state.cash
            total_needed = state.cash
            guard_active = True
            if total_needed <= 0:
                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, event_type="Rebalance-A",
                    ticker=t, action="BUY",
                    shares_before=state.positions.get(t, 0.0),
                    shares_after=state.positions.get(t, 0.0),
                    share_delta=0.0, execution_price=price,
                    gross_trade_value=0.0, transaction_cost=0.0,
                    net_cash_impact=0.0,
                    cash_before=state.cash, cash_after=state.cash,
                    funding_source="Insufficient cash", paired_trade_id=pair_id,
                    guard_active=True, shortfall_amount=shortfall,
                    note=f"BLOCKED: needed ${shortfall:.2f} more cash",
                ))
                continue

        cost = total_needed * (tx_cost_pct / 100.0) / (1 + tx_cost_pct / 100.0)
        net_buy = total_needed - cost
        shares_buy = net_buy / price

        cash_before = state.cash
        shares_before = state.positions.get(t, 0.0)
        state.positions[t] = shares_before + shares_buy
        state.cash -= total_needed

        trades.append(TradeRecord(
            date=date, sleeve_name=sleeve.name, event_type="Rebalance-A",
            ticker=t, action="BUY",
            shares_before=shares_before, shares_after=state.positions[t],
            share_delta=shares_buy, execution_price=price,
            gross_trade_value=net_buy, transaction_cost=cost,
            net_cash_impact=-total_needed,
            cash_before=cash_before, cash_after=state.cash,
            funding_source="SELL proceeds" if sells else "Sleeve cash",
            paired_trade_id=pair_id,
            guard_active=guard_active, shortfall_amount=shortfall,
        ))

    # Conservation check
    value_after = state.total_value(prices)
    _assert_conservation("Mode-A", value_before, value_after)

    return trades


# ---------------------------------------------------------------------------
# Mode B: Signal-Based (Kelly 3% Rule)
# ---------------------------------------------------------------------------

def rebalance_mode_b(
    state: SleeveState,
    prices: dict[str, float],
    date: dt.date,
    tx_cost_pct: float,
    contribution_cash: float = 0.0,
) -> list[TradeRecord]:
    """Perform Mode B signal-based rebalancing with double-entry cash routing.

    Both sides of each trade are logged. All proceeds route through cash.
    """
    trades: list[TradeRecord] = []
    sleeve = state.sleeve
    state.cash += contribution_cash

    stock_ticker = sleeve.stock_ticker
    bond_ticker = sleeve.bond_ticker
    if not stock_ticker or not bond_ticker:
        return trades

    stock_price = prices.get(stock_ticker, 0.0)
    bond_price = prices.get(bond_ticker, 0.0)
    if stock_price <= 0 or bond_price <= 0:
        return trades

    stock_shares = state.positions.get(stock_ticker, 0.0)
    bond_shares = state.positions.get(bond_ticker, 0.0)
    stock_value = stock_shares * stock_price
    bond_value = bond_shares * bond_price

    value_before = state.total_value(prices)

    # Compute signal
    annual_yield = sleeve.annualized_yield_pct / 100.0
    period_frac = frequency_to_annual_fraction(sleeve.signal_frequency)
    signal = annual_yield * stock_value * period_frac * sleeve.signal_scaling

    # Compare stock value change vs signal
    prev_stock = state.prev_stock_value if state.prev_stock_value > 0 else stock_value
    change = stock_value - prev_stock

    direction = "HOLD"
    trade_amount = 0.0

    if change >= signal and signal > 0:
        direction = "SELL"  # SELL stock, BUY bond
        trade_amount = min(signal, stock_value)
    elif change <= -signal and signal > 0:
        direction = "BUY"  # BUY stock, SELL bond
        trade_amount = min(signal, bond_value + state.cash)
    elif sleeve.force_sideways_trade and signal > 0:
        if change >= 0:
            direction = "SELL"
            trade_amount = min(signal * 0.5, stock_value)
        else:
            direction = "BUY"
            trade_amount = min(signal * 0.5, bond_value + state.cash)

    if trade_amount < 1.0:
        direction = "HOLD"

    pair_id = _new_pair_id()

    if direction == "SELL":
        # LEG 1: SELL stock → cash
        sell_shares = min(trade_amount / stock_price, stock_shares)
        if sell_shares <= 0:
            state.prev_stock_value = state.positions.get(stock_ticker, 0.0) * stock_price
            return trades

        gross_sell = sell_shares * stock_price
        sell_cost = gross_sell * (tx_cost_pct / 100.0)
        sell_proceeds = gross_sell - sell_cost

        cash_before = state.cash
        shares_before_stock = state.positions.get(stock_ticker, 0.0)
        state.positions[stock_ticker] = shares_before_stock - sell_shares
        state.cash += sell_proceeds

        trades.append(TradeRecord(
            date=date, sleeve_name=sleeve.name, event_type="Rebalance-B",
            ticker=stock_ticker, action="SELL",
            shares_before=shares_before_stock,
            shares_after=state.positions[stock_ticker],
            share_delta=-sell_shares, execution_price=stock_price,
            gross_trade_value=gross_sell, transaction_cost=sell_cost,
            net_cash_impact=sell_proceeds,
            cash_before=cash_before, cash_after=state.cash,
            funding_source="Position liquidation", paired_trade_id=pair_id,
            signal_amount=signal,
        ))

        # LEG 2: BUY bond ← cash
        buy_cost = sell_proceeds * (tx_cost_pct / 100.0)
        net_for_buy = sell_proceeds - buy_cost

        # Cash guard
        guard_active = False
        shortfall = 0.0
        actual_spend = sell_proceeds
        if actual_spend > state.cash:
            shortfall = actual_spend - state.cash
            actual_spend = state.cash
            guard_active = True
            buy_cost = actual_spend * (tx_cost_pct / 100.0)
            net_for_buy = actual_spend - buy_cost

        if net_for_buy > 0 and bond_price > 0:
            buy_shares = net_for_buy / bond_price
            cash_before2 = state.cash
            shares_before_bond = state.positions.get(bond_ticker, 0.0)
            state.positions[bond_ticker] = shares_before_bond + buy_shares
            state.cash -= actual_spend

            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-B",
                ticker=bond_ticker, action="BUY",
                shares_before=shares_before_bond,
                shares_after=state.positions[bond_ticker],
                share_delta=buy_shares, execution_price=bond_price,
                gross_trade_value=net_for_buy, transaction_cost=buy_cost,
                net_cash_impact=-actual_spend,
                cash_before=cash_before2, cash_after=state.cash,
                funding_source="SELL proceeds", paired_trade_id=pair_id,
                signal_amount=signal, guard_active=guard_active,
                shortfall_amount=shortfall,
            ))

    elif direction == "BUY":
        # LEG 1: SELL bond → cash
        bond_sell_val = min(trade_amount, bond_value)
        sell_b_shares = bond_sell_val / bond_price if bond_price > 0 else 0.0
        sell_b_shares = min(sell_b_shares, bond_shares)

        if sell_b_shares > 0:
            gross_sell = sell_b_shares * bond_price
            sell_cost = gross_sell * (tx_cost_pct / 100.0)
            sell_proceeds = gross_sell - sell_cost

            cash_before = state.cash
            shares_before_bond = state.positions.get(bond_ticker, 0.0)
            state.positions[bond_ticker] = shares_before_bond - sell_b_shares
            state.cash += sell_proceeds

            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-B",
                ticker=bond_ticker, action="SELL",
                shares_before=shares_before_bond,
                shares_after=state.positions[bond_ticker],
                share_delta=-sell_b_shares, execution_price=bond_price,
                gross_trade_value=gross_sell, transaction_cost=sell_cost,
                net_cash_impact=sell_proceeds,
                cash_before=cash_before, cash_after=state.cash,
                funding_source="Position liquidation", paired_trade_id=pair_id,
                signal_amount=signal,
            ))

        # LEG 2: BUY stock ← cash
        available_for_buy = state.cash
        buy_cost = available_for_buy * (tx_cost_pct / 100.0)
        net_for_buy = available_for_buy - buy_cost

        # Limit to trade_amount
        desired_spend = trade_amount
        guard_active = False
        shortfall = 0.0
        if desired_spend > available_for_buy:
            shortfall = desired_spend - available_for_buy
            desired_spend = available_for_buy
            guard_active = True

        buy_cost = desired_spend * (tx_cost_pct / 100.0)
        net_for_buy = desired_spend - buy_cost

        if net_for_buy > 0 and stock_price > 0:
            buy_s_shares = net_for_buy / stock_price
            cash_before2 = state.cash
            shares_before_stock = state.positions.get(stock_ticker, 0.0)
            state.positions[stock_ticker] = shares_before_stock + buy_s_shares
            state.cash -= desired_spend

            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-B",
                ticker=stock_ticker, action="BUY",
                shares_before=shares_before_stock,
                shares_after=state.positions[stock_ticker],
                share_delta=buy_s_shares, execution_price=stock_price,
                gross_trade_value=net_for_buy, transaction_cost=buy_cost,
                net_cash_impact=-desired_spend,
                cash_before=cash_before2, cash_after=state.cash,
                funding_source="SELL proceeds", paired_trade_id=pair_id,
                signal_amount=signal, guard_active=guard_active,
                shortfall_amount=shortfall,
            ))

    else:
        # HOLD
        trades.append(TradeRecord(
            date=date, sleeve_name=sleeve.name, event_type="Rebalance-B",
            ticker=stock_ticker, action="HOLD",
            shares_before=stock_shares, shares_after=stock_shares,
            share_delta=0.0, execution_price=stock_price,
            cash_before=state.cash, cash_after=state.cash,
            signal_amount=signal, note="Sideways",
        ))

    # Update prev stock value for next period
    state.prev_stock_value = state.positions.get(stock_ticker, 0.0) * stock_price

    # Conservation check
    value_after = state.total_value(prices)
    _assert_conservation("Mode-B", value_before, value_after)

    return trades


# ---------------------------------------------------------------------------
# Mode C: Custom Formula
# ---------------------------------------------------------------------------

def rebalance_mode_c(
    state: SleeveState,
    prices: dict[str, float],
    date: dt.date,
    config: PortfolioConfig,
    tx_cost_pct: float,
    contribution_cash: float = 0.0,
    yield_data: Optional[dict[str, float]] = None,
) -> list[TradeRecord]:
    """Evaluate custom formula and apply double-entry trade logic.

    *yield_data* is an optional ``{ticker: yield_decimal}`` mapping that
    supplies real distribution yields for the ``portfolio["X"].yield_ttm``
    variable in the formula sandbox.
    """
    trades: list[TradeRecord] = []
    sleeve = state.sleeve
    state.cash += contribution_cash

    if not sleeve.custom_formula.strip():
        return trades

    if yield_data is None:
        yield_data = {}

    # Build sandboxed evaluation context
    aeval = Interpreter()

    class _HoldingProxy:
        def __init__(self, ticker: str):
            self.ticker = ticker
            self.price = prices.get(ticker, 0.0)
            self.shares = state.positions.get(ticker, 0.0)
            self.value = self.price * self.shares
            self.yield_ttm = yield_data.get(ticker, 0.0)

    class _PortfolioProxy:
        def __init__(self):
            self.total_value = config.total_value
            self._holdings = {
                h.ticker: _HoldingProxy(h.ticker)
                for h in config.holdings if h.ticker
            }

        def __getitem__(self, key):
            return self._holdings.get(key, _HoldingProxy(key))

    aeval.symtable["portfolio"] = _PortfolioProxy()
    aeval.symtable["date"] = date
    aeval.symtable["cash"] = state.cash

    try:
        result = aeval(sleeve.custom_formula)
        if result is None or not isinstance(result, (int, float)):
            return trades
        signal = float(result)
    except Exception:
        return trades

    # Apply signal: positive = buy first ticker / sell second; negative = reverse
    tickers = [
        config.holding_by_id(hid).ticker
        for hid in sleeve.holding_ids
        if config.holding_by_id(hid) and config.holding_by_id(hid).ticker
    ]
    if len(tickers) < 2:
        return trades

    primary = tickers[0]
    secondary = tickers[1]

    primary_price = prices.get(primary, 0.0)
    secondary_price = prices.get(secondary, 0.0)
    if primary_price <= 0 or secondary_price <= 0:
        return trades

    abs_signal = abs(signal)
    if abs_signal < 1.0:
        return trades

    value_before = state.total_value(prices)
    pair_id = _new_pair_id()

    if signal > 0:
        # BUY primary, SELL secondary
        # LEG 1: SELL secondary → cash
        sec_shares = state.positions.get(secondary, 0.0)
        sec_val = sec_shares * secondary_price
        sell_val = min(abs_signal, sec_val)
        sell_shares = sell_val / secondary_price if secondary_price > 0 else 0.0
        sell_shares = min(sell_shares, sec_shares)

        if sell_shares > 0:
            gross_sell = sell_shares * secondary_price
            sell_cost = gross_sell * (tx_cost_pct / 100.0)
            sell_proceeds = gross_sell - sell_cost

            cash_before = state.cash
            shares_before = state.positions.get(secondary, 0.0)
            state.positions[secondary] = shares_before - sell_shares
            state.cash += sell_proceeds

            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-C",
                ticker=secondary, action="SELL",
                shares_before=shares_before,
                shares_after=state.positions[secondary],
                share_delta=-sell_shares, execution_price=secondary_price,
                gross_trade_value=gross_sell, transaction_cost=sell_cost,
                net_cash_impact=sell_proceeds,
                cash_before=cash_before, cash_after=state.cash,
                funding_source="Position liquidation", paired_trade_id=pair_id,
                signal_amount=signal,
            ))

        # LEG 2: BUY primary ← cash
        desired_spend = min(abs_signal, state.cash)
        guard_active = False
        shortfall = 0.0
        if abs_signal > state.cash:
            shortfall = abs_signal - state.cash
            guard_active = True

        if desired_spend > 0:
            buy_cost = desired_spend * (tx_cost_pct / 100.0)
            net_for_buy = desired_spend - buy_cost

            if net_for_buy > 0 and primary_price > 0:
                buy_shares = net_for_buy / primary_price
                cash_before2 = state.cash
                shares_before2 = state.positions.get(primary, 0.0)
                state.positions[primary] = shares_before2 + buy_shares
                state.cash -= desired_spend

                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, event_type="Rebalance-C",
                    ticker=primary, action="BUY",
                    shares_before=shares_before2,
                    shares_after=state.positions[primary],
                    share_delta=buy_shares, execution_price=primary_price,
                    gross_trade_value=net_for_buy, transaction_cost=buy_cost,
                    net_cash_impact=-desired_spend,
                    cash_before=cash_before2, cash_after=state.cash,
                    funding_source="SELL proceeds", paired_trade_id=pair_id,
                    signal_amount=signal, guard_active=guard_active,
                    shortfall_amount=shortfall,
                ))

    else:
        # SELL primary, BUY secondary
        # LEG 1: SELL primary → cash
        pri_shares = state.positions.get(primary, 0.0)
        pri_val = pri_shares * primary_price
        sell_val = min(abs_signal, pri_val)
        sell_shares = sell_val / primary_price if primary_price > 0 else 0.0
        sell_shares = min(sell_shares, pri_shares)

        if sell_shares > 0:
            gross_sell = sell_shares * primary_price
            sell_cost = gross_sell * (tx_cost_pct / 100.0)
            sell_proceeds = gross_sell - sell_cost

            cash_before = state.cash
            shares_before = state.positions.get(primary, 0.0)
            state.positions[primary] = shares_before - sell_shares
            state.cash += sell_proceeds

            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, event_type="Rebalance-C",
                ticker=primary, action="SELL",
                shares_before=shares_before,
                shares_after=state.positions[primary],
                share_delta=-sell_shares, execution_price=primary_price,
                gross_trade_value=gross_sell, transaction_cost=sell_cost,
                net_cash_impact=sell_proceeds,
                cash_before=cash_before, cash_after=state.cash,
                funding_source="Position liquidation", paired_trade_id=pair_id,
                signal_amount=signal,
            ))

        # LEG 2: BUY secondary ← cash
        desired_spend = min(abs_signal, state.cash)
        guard_active = False
        shortfall = 0.0
        if abs_signal > state.cash:
            shortfall = abs_signal - state.cash
            guard_active = True

        if desired_spend > 0:
            buy_cost = desired_spend * (tx_cost_pct / 100.0)
            net_for_buy = desired_spend - buy_cost

            if net_for_buy > 0 and secondary_price > 0:
                buy_shares = net_for_buy / secondary_price
                cash_before2 = state.cash
                shares_before2 = state.positions.get(secondary, 0.0)
                state.positions[secondary] = shares_before2 + buy_shares
                state.cash -= desired_spend

                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, event_type="Rebalance-C",
                    ticker=secondary, action="BUY",
                    shares_before=shares_before2,
                    shares_after=state.positions[secondary],
                    share_delta=buy_shares, execution_price=secondary_price,
                    gross_trade_value=net_for_buy, transaction_cost=buy_cost,
                    net_cash_impact=-desired_spend,
                    cash_before=cash_before2, cash_after=state.cash,
                    funding_source="SELL proceeds", paired_trade_id=pair_id,
                    signal_amount=signal, guard_active=guard_active,
                    shortfall_amount=shortfall,
                ))

    # Conservation check
    value_after = state.total_value(prices)
    _assert_conservation("Mode-C", value_before, value_after)

    return trades


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_rebalancing(
    state: SleeveState,
    prices: dict[str, float],
    date: dt.date,
    config: PortfolioConfig,
    contribution_cash: float = 0.0,
    yield_data: Optional[dict[str, float]] = None,
) -> list[TradeRecord]:
    """Dispatch to the correct rebalancing mode for this sleeve."""
    sleeve = state.sleeve
    mode = sleeve.mode
    tx = sleeve.transaction_cost_pct

    if mode.startswith("A"):
        trigger = sleeve.rebalance_trigger
        if trigger in ("Monthly", "Quarterly", "Semi-Annually", "Annually"):
            trigger_type = "Calendar"
        elif trigger == "Threshold":
            trigger_type = "Threshold"
        else:
            trigger_type = "Hybrid"
        return rebalance_mode_a(
            state, prices, date, config, trigger_type,
            sleeve.threshold_pct, sleeve.rebalance_method, tx,
            contribution_cash,
        )
    elif mode.startswith("B"):
        return rebalance_mode_b(state, prices, date, tx, contribution_cash)
    elif mode.startswith("C"):
        return rebalance_mode_c(
            state, prices, date, config, tx, contribution_cash,
            yield_data=yield_data,
        )

    return []
