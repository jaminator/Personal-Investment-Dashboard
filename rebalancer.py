"""
rebalancer.py — Multi-sleeve rebalancing engine (Modes A, B, C)
with dividend event tracking (DRIP / Cash accrual).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from asteval import Interpreter

from contributions import frequency_to_annual_fraction, rebalance_dates
from portfolio import Sleeve, PortfolioConfig


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date: dt.date = dt.date.today()
    sleeve_name: str = ""
    ticker: str = ""
    direction: str = ""  # BUY / SELL / HOLD
    shares: float = 0.0
    dollar_amount: float = 0.0
    signal_amount: float = 0.0
    transaction_cost: float = 0.0
    note: str = ""


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
    # Pending DRIP amounts: ticker → [(payment_date, dollar_amount), ...]
    pending_drip: dict[str, list] = field(default_factory=dict)

    def position_value(self, prices: dict[str, float]) -> dict[str, float]:
        """Dollar value per ticker."""
        return {t: self.positions.get(t, 0.0) * prices.get(t, 0.0)
                for t in self.positions}

    def total_value(self, prices: dict[str, float]) -> float:
        return sum(self.position_value(prices).values()) + self.cash


# ---------------------------------------------------------------------------
# Mode A: Target-Weight Rebalancing
# ---------------------------------------------------------------------------

def rebalance_mode_a(
    state: SleeveState,
    prices: dict[str, float],
    date: dt.date,
    trigger_type: str,
    threshold_pct: float,
    method: str,
    tx_cost_pct: float,
    contribution_cash: float = 0.0,
) -> list[TradeRecord]:
    """Perform Mode A rebalancing. Returns list of trade records."""
    trades: list[TradeRecord] = []
    sleeve = state.sleeve
    total = state.total_value(prices) + contribution_cash
    if total <= 0:
        return trades

    state.cash += contribution_cash
    total = state.total_value(prices)
    if total <= 0:
        return trades

    targets = sleeve.target_weights
    if not targets:
        return trades

    # Current weights within the sleeve
    pos_vals = state.position_value(prices)
    current_weights = {t: v / total if total > 0 else 0.0 for t, v in pos_vals.items()}

    # Check threshold trigger
    if trigger_type in ("Threshold", "Hybrid"):
        max_drift = max(
            abs(current_weights.get(t, 0.0) - targets.get(t, 0.0) / 100.0)
            for t in targets
        )
        if max_drift * 100 < threshold_pct:
            return trades

    # Compute target values
    target_vals = {t: total * (w / 100.0) for t, w in targets.items()}

    if method == "Contributions-only":
        # Only deploy cash to underweight positions
        available = state.cash
        if available <= 0:
            return trades
        underweight = {
            t: max(0, target_vals.get(t, 0.0) - pos_vals.get(t, 0.0))
            for t in targets
        }
        total_uw = sum(underweight.values())
        if total_uw <= 0:
            return trades
        for t, uw in underweight.items():
            if uw <= 0:
                continue
            alloc = available * (uw / total_uw)
            cost = alloc * (tx_cost_pct / 100.0)
            net = alloc - cost
            if net <= 0 or prices.get(t, 0) <= 0:
                continue
            shares_buy = net / prices[t]
            state.positions[t] = state.positions.get(t, 0.0) + shares_buy
            state.cash -= alloc
            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, ticker=t,
                direction="BUY", shares=shares_buy, dollar_amount=net,
                transaction_cost=cost, note="Contributions-only rebalance",
            ))
        return trades

    # Full or Partial rebalancing
    for t, tw in targets.items():
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

        cost = abs(diff) * (tx_cost_pct / 100.0)
        if diff > 0:
            # BUY
            net_buy = diff - cost
            if net_buy <= 0:
                continue
            shares_buy = net_buy / price
            state.positions[t] = state.positions.get(t, 0.0) + shares_buy
            state.cash -= diff
            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, ticker=t,
                direction="BUY", shares=shares_buy, dollar_amount=net_buy,
                transaction_cost=cost,
            ))
        else:
            # SELL
            shares_sell = min(abs(diff) / price, state.positions.get(t, 0.0))
            if shares_sell <= 0:
                continue
            proceeds = shares_sell * price - cost
            state.positions[t] = state.positions.get(t, 0.0) - shares_sell
            state.cash += proceeds
            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, ticker=t,
                direction="SELL", shares=shares_sell, dollar_amount=proceeds,
                transaction_cost=cost,
            ))

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
    """Perform Mode B signal-based rebalancing."""
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
        # Stock went UP → SELL stock, BUY bond
        direction = "SELL"
        trade_amount = min(signal, stock_value)
    elif change <= -signal and signal > 0:
        # Stock went DOWN → BUY stock, SELL bond
        direction = "BUY"
        trade_amount = min(signal, bond_value + state.cash)
    elif sleeve.force_sideways_trade and signal > 0:
        # Force minimum trade in sideways market
        if change >= 0:
            direction = "SELL"
            trade_amount = min(signal * 0.5, stock_value)
        else:
            direction = "BUY"
            trade_amount = min(signal * 0.5, bond_value + state.cash)

    if trade_amount < 1.0:
        direction = "HOLD"

    if direction == "SELL":
        # Sell stock, buy bond
        cost = trade_amount * (tx_cost_pct / 100.0)
        sell_shares = min(trade_amount / stock_price, stock_shares)
        if sell_shares > 0:
            net = sell_shares * stock_price - cost
            state.positions[stock_ticker] = stock_shares - sell_shares
            if net > 0 and bond_price > 0:
                buy_shares = (net - net * tx_cost_pct / 100.0) / bond_price
                state.positions[bond_ticker] = bond_shares + buy_shares
                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, ticker=stock_ticker,
                    direction="SELL", shares=sell_shares,
                    dollar_amount=sell_shares * stock_price,
                    signal_amount=signal, transaction_cost=cost * 2,
                ))
    elif direction == "BUY":
        # Sell bond, buy stock
        cost = trade_amount * (tx_cost_pct / 100.0)
        # First use bond holdings, then cash
        bond_sell_val = min(trade_amount, bond_value)
        cash_needed = trade_amount - bond_sell_val
        sell_b_shares = bond_sell_val / bond_price if bond_price > 0 else 0.0
        if sell_b_shares > 0:
            state.positions[bond_ticker] = max(0, bond_shares - sell_b_shares)
        if cash_needed > 0:
            cash_use = min(cash_needed, state.cash)
            state.cash -= cash_use
            bond_sell_val += cash_use
        net = bond_sell_val - cost * 2
        if net > 0 and stock_price > 0:
            buy_s_shares = net / stock_price
            state.positions[stock_ticker] = stock_shares + buy_s_shares
            trades.append(TradeRecord(
                date=date, sleeve_name=sleeve.name, ticker=stock_ticker,
                direction="BUY", shares=buy_s_shares,
                dollar_amount=buy_s_shares * stock_price,
                signal_amount=signal, transaction_cost=cost * 2,
            ))
    else:
        trades.append(TradeRecord(
            date=date, sleeve_name=sleeve.name, ticker=stock_ticker,
            direction="HOLD", signal_amount=signal, note="Sideways",
        ))

    # Update prev stock value for next period
    state.prev_stock_value = state.positions.get(stock_ticker, 0.0) * stock_price

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
    """Evaluate custom formula and apply signal-based trade logic.

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

    # Apply signal like Mode B: positive = buy first ticker, negative = sell first ticker
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

    cost = abs_signal * (tx_cost_pct / 100.0)

    if signal > 0:
        # BUY primary, SELL secondary
        sec_shares = state.positions.get(secondary, 0.0)
        sec_val = sec_shares * secondary_price
        trade_val = min(abs_signal, sec_val + state.cash)
        if trade_val > 0:
            sell_sec = min(trade_val, sec_val)
            sell_sec_shares = sell_sec / secondary_price if secondary_price > 0 else 0
            state.positions[secondary] = max(0, sec_shares - sell_sec_shares)
            cash_use = min(trade_val - sell_sec, state.cash)
            state.cash -= cash_use
            net = trade_val - cost * 2
            if net > 0:
                buy_shares = net / primary_price
                state.positions[primary] = state.positions.get(primary, 0.0) + buy_shares
                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, ticker=primary,
                    direction="BUY", shares=buy_shares,
                    dollar_amount=net, signal_amount=signal,
                    transaction_cost=cost * 2,
                ))
    else:
        # SELL primary, BUY secondary
        pri_shares = state.positions.get(primary, 0.0)
        pri_val = pri_shares * primary_price
        trade_val = min(abs_signal, pri_val)
        if trade_val > 0:
            sell_shares = trade_val / primary_price if primary_price > 0 else 0
            state.positions[primary] = max(0, pri_shares - sell_shares)
            net = trade_val - cost * 2
            if net > 0 and secondary_price > 0:
                buy_shares = net / secondary_price
                state.positions[secondary] = state.positions.get(secondary, 0.0) + buy_shares
                trades.append(TradeRecord(
                    date=date, sleeve_name=sleeve.name, ticker=primary,
                    direction="SELL", shares=sell_shares,
                    dollar_amount=trade_val, signal_amount=signal,
                    transaction_cost=cost * 2,
                ))

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
            state, prices, date, trigger_type,
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
