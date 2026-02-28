"""
portfolio.py — Portfolio configuration models and helpers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Asset classes
# ---------------------------------------------------------------------------

ASSET_CLASSES = [
    "Equity",
    "Fixed Income",
    "Real Estate",
    "Commodity",
    "Cash",
    "Alternative",
    "Options",
]

FREQUENCIES = [
    "Daily",
    "Weekly",
    "Bi-Weekly",
    "Semi-Monthly",
    "Monthly",
    "Quarterly",
    "Semi-Annually",
    "Annually",
]

REBALANCE_MODES = ["A: Target-Weight", "B: Signal-Based (Kelly)", "C: Custom Formula"]


# ---------------------------------------------------------------------------
# Holding
# ---------------------------------------------------------------------------

@dataclass
class Holding:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ticker: str = ""
    shares: float = 0.0
    cost_basis: float = 0.0
    asset_class: str = "Equity"
    is_manual: bool = False
    manual_value: float = 0.0
    manual_expected_return: float = 0.07
    current_price: float = 0.0

    # Dividend settings
    drip_enabled: bool = True  # True = reinvest, False = accrue to cash
    payment_date_offset: int = 20  # calendar days after ex-date (fallback)
    distributions_per_year_override: int = 0  # 0 = auto-detect

    @property
    def market_value(self) -> float:
        if self.is_manual:
            return self.manual_value
        return self.shares * self.current_price

    @property
    def total_cost(self) -> float:
        return self.shares * self.cost_basis

    @property
    def unrealized_gl(self) -> float:
        return self.market_value - self.total_cost

    @property
    def unrealized_gl_pct(self) -> float:
        cost = self.total_cost
        if cost == 0:
            return 0.0
        return self.unrealized_gl / cost

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Holding":
        d.pop("market_value", None)
        d.pop("total_cost", None)
        d.pop("unrealized_gl", None)
        d.pop("unrealized_gl_pct", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Contribution stream
# ---------------------------------------------------------------------------

@dataclass
class ContributionStream:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    amount: float = 500.0
    frequency: str = "Monthly"
    start_date: str = ""
    end_date: str = ""
    allocation_mode: str = "Proportional"  # Proportional | Specific Ticker | Cash
    target_ticker: str = ""
    target_sleeve: str = ""  # empty = proportional across sleeves

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContributionStream":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Withdrawal
# ---------------------------------------------------------------------------

@dataclass
class Withdrawal:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    date: str = ""
    amount: float = 0.0
    is_percentage: bool = False
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Withdrawal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Sleeve
# ---------------------------------------------------------------------------

@dataclass
class Sleeve:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "New Sleeve"
    holding_ids: list[str] = field(default_factory=list)
    mode: str = "A: Target-Weight"
    target_weights: dict[str, float] = field(default_factory=dict)

    # Mode A settings
    rebalance_trigger: str = "Quarterly"  # Calendar / Threshold / Hybrid
    threshold_pct: float = 5.0
    rebalance_method: str = "Full"  # Full / Contributions-only / Partial
    transaction_cost_pct: float = 0.1

    # Mode B settings
    bond_ticker: str = ""
    stock_ticker: str = ""
    annualized_yield_pct: float = 12.0
    signal_scaling: float = 1.0
    signal_frequency: str = "Monthly"
    force_sideways_trade: bool = False

    # Mode C settings
    custom_formula: str = ""

    cash_balance: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sleeve":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Portfolio Config
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    holdings: list[Holding] = field(default_factory=list)
    contributions: list[ContributionStream] = field(default_factory=list)
    withdrawals: list[Withdrawal] = field(default_factory=list)
    sleeves: list[Sleeve] = field(default_factory=list)
    cross_sleeve_cash_pooling: bool = False

    # Simulation settings
    forecast_years: int = 10
    num_simulations: int = 1000
    simulation_method: str = "GBM"  # GBM / Bootstrap
    return_source: str = "Historical"  # Historical / Manual
    lookback_period: str = "5Y"
    manual_return: float = 0.08
    manual_volatility: float = 0.15
    target_value: float = 1_000_000.0

    # Backtest settings
    backtest_start: str = ""
    backtest_end: str = ""
    benchmark_tickers: list[str] = field(default_factory=lambda: ["SPY"])

    @property
    def total_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings if h.ticker and not h.is_manual]

    def weights(self) -> dict[str, float]:
        tv = self.total_value
        if tv == 0:
            return {}
        return {h.ticker: h.market_value / tv for h in self.holdings if h.ticker}

    def holding_by_id(self, hid: str) -> Optional[Holding]:
        for h in self.holdings:
            if h.id == hid:
                return h
        return None

    def holding_by_ticker(self, ticker: str) -> Optional[Holding]:
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    def sleeve_for_holding(self, hid: str) -> Optional[str]:
        for s in self.sleeves:
            if hid in s.holding_ids:
                return s.id
        return None

    def unassigned_holdings(self) -> list[Holding]:
        assigned = set()
        for s in self.sleeves:
            assigned.update(s.holding_ids)
        return [h for h in self.holdings if h.id not in assigned]

    # --- Serialisation -------------------------------------------------------

    def to_json(self) -> str:
        d = {
            "holdings": [h.to_dict() for h in self.holdings],
            "contributions": [c.to_dict() for c in self.contributions],
            "withdrawals": [w.to_dict() for w in self.withdrawals],
            "sleeves": [s.to_dict() for s in self.sleeves],
            "cross_sleeve_cash_pooling": self.cross_sleeve_cash_pooling,
            "forecast_years": self.forecast_years,
            "num_simulations": self.num_simulations,
            "simulation_method": self.simulation_method,
            "return_source": self.return_source,
            "lookback_period": self.lookback_period,
            "manual_return": self.manual_return,
            "manual_volatility": self.manual_volatility,
            "target_value": self.target_value,
            "backtest_start": self.backtest_start,
            "backtest_end": self.backtest_end,
            "benchmark_tickers": self.benchmark_tickers,
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PortfolioConfig":
        d = json.loads(text)
        pc = cls()
        pc.holdings = [Holding.from_dict(h) for h in d.get("holdings", [])]
        pc.contributions = [ContributionStream.from_dict(c) for c in d.get("contributions", [])]
        pc.withdrawals = [Withdrawal.from_dict(w) for w in d.get("withdrawals", [])]
        pc.sleeves = [Sleeve.from_dict(s) for s in d.get("sleeves", [])]
        for key in [
            "cross_sleeve_cash_pooling", "forecast_years", "num_simulations",
            "simulation_method", "return_source", "lookback_period",
            "manual_return", "manual_volatility", "target_value",
            "backtest_start", "backtest_end", "benchmark_tickers",
        ]:
            if key in d:
                setattr(pc, key, d[key])
        return pc


# ---------------------------------------------------------------------------
# Default portfolio
# ---------------------------------------------------------------------------

def default_portfolio() -> PortfolioConfig:
    """VO / PDI sample portfolio with a Modified Kelly sleeve."""
    pc = PortfolioConfig()

    # Current prices (as of Feb 2026)
    vo_price = 307.06
    pdi_price = 18.13

    # PDI trailing dividend yield: $2.65 annual / $18.13 ≈ 14.62%
    pdi_yield = 2.65 / pdi_price

    # Weight formula: PDI weight = 12% / (12% + PDI yield)
    pdi_weight = 0.12 / (0.12 + pdi_yield)
    vo_weight = 1.0 - pdi_weight

    total_value = 300_000.0

    vo_shares = round(total_value * vo_weight / vo_price, 2)
    pdi_shares = round(total_value * pdi_weight / pdi_price, 2)

    vo = Holding(
        ticker="VO", shares=vo_shares, cost_basis=vo_price,
        asset_class="Equity",
    )
    pdi = Holding(
        ticker="PDI", shares=pdi_shares, cost_basis=pdi_price,
        asset_class="Fixed Income",
    )
    pc.holdings = [vo, pdi]

    # Modified Kelly sleeve — PDI as debt, VO as equity, quarterly rebalancing
    sleeve = Sleeve(
        name="Modified Kelly",
        holding_ids=[vo.id, pdi.id],
        mode="B: Signal-Based (Kelly)",
        bond_ticker="PDI",
        stock_ticker="VO",
        signal_frequency="Quarterly",
    )
    pc.sleeves = [sleeve]

    pc.contributions = []
    return pc
