"""
analytics.py — Comprehensive risk, return, and risk-adjusted metrics.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

from data_fetcher import (
    annualized_return_vol,
    cagr,
    correlation_matrix,
    fetch_risk_free_rate,
    fetch_returns,
)


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def total_return(series: pd.Series) -> float:
    """Cumulative total return from a price or value series."""
    if len(series) < 2:
        return 0.0
    return float(series.iloc[-1] / series.iloc[0] - 1)


def annualized_cagr(series: pd.Series, trading_days: int = 252) -> float:
    """CAGR from a price series."""
    if len(series) < 2:
        return 0.0
    total = series.iloc[-1] / series.iloc[0]
    n_years = len(series) / trading_days
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def best_worst_year(returns: pd.Series) -> dict:
    """Best and worst calendar year returns."""
    if returns.empty:
        return {"best_year": 0, "best_year_ret": 0, "worst_year": 0, "worst_year_ret": 0}
    annual = (1 + returns).resample("YE").prod() - 1
    if annual.empty:
        return {"best_year": 0, "best_year_ret": 0, "worst_year": 0, "worst_year_ret": 0}
    return {
        "best_year": int(annual.idxmax().year) if not annual.empty else 0,
        "best_year_ret": float(annual.max()),
        "worst_year": int(annual.idxmin().year) if not annual.empty else 0,
        "worst_year_ret": float(annual.min()),
    }


def best_worst_month(returns: pd.Series) -> dict:
    """Best and worst calendar month returns."""
    if returns.empty:
        return {"best_month": "", "best_month_ret": 0, "worst_month": "", "worst_month_ret": 0}
    monthly = (1 + returns).resample("ME").prod() - 1
    if monthly.empty:
        return {"best_month": "", "best_month_ret": 0, "worst_month": "", "worst_month_ret": 0}
    return {
        "best_month": str(monthly.idxmax().strftime("%Y-%m")),
        "best_month_ret": float(monthly.max()),
        "worst_month": str(monthly.idxmin().strftime("%Y-%m")),
        "worst_month_ret": float(monthly.min()),
    }


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))


def max_drawdown(series: pd.Series) -> tuple[float, str, str]:
    """Return (max_dd_fraction, peak_date, trough_date)."""
    if len(series) < 2:
        return 0.0, "", ""
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    trough_idx = drawdown.idxmin()
    peak_idx = series.loc[:trough_idx].idxmax()
    dd = float(drawdown.min())
    return dd, str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx), \
           str(trough_idx.date()) if hasattr(trough_idx, 'date') else str(trough_idx)


def drawdown_series(series: pd.Series) -> pd.Series:
    """Underwater chart: drawdown fraction over time."""
    if len(series) < 2:
        return pd.Series(dtype=float)
    cummax = series.cummax()
    return (series - cummax) / cummax


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk."""
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR."""
    if returns.empty:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    return float(mu + sigma * stats.norm.ppf(1 - confidence))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall."""
    if returns.empty:
        return 0.0
    var = var_historical(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free: float = None, trading_days: int = 252) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    daily_rf = risk_free / trading_days
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, risk_free: float = None, trading_days: int = 252) -> float:
    if returns.empty:
        return 0.0
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    daily_rf = risk_free / trading_days
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(trading_days))


def calmar_ratio(series: pd.Series, trading_days: int = 252) -> float:
    """CAGR / |Max Drawdown|."""
    c = annualized_cagr(series, trading_days)
    dd, _, _ = max_drawdown(series)
    if dd == 0:
        return 0.0
    return float(c / abs(dd))


def beta_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = None,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """Return (beta, alpha, r_squared) vs. benchmark."""
    if returns.empty or benchmark_returns.empty:
        return 0.0, 0.0, 0.0
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    daily_rf = risk_free / trading_days
    aligned = pd.DataFrame({"port": returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0, 0.0
    excess_port = aligned["port"] - daily_rf
    excess_bench = aligned["bench"] - daily_rf
    slope, intercept, r_value, _, _ = stats.linregress(excess_bench, excess_port)
    beta = float(slope)
    alpha = float(intercept * trading_days)  # annualize
    r_squared = float(r_value ** 2)
    return beta, alpha, r_squared


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = None,
    trading_days: int = 252,
) -> float:
    """Treynor ratio = (portfolio return - risk-free) / beta."""
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    b, _, _ = beta_alpha(returns, benchmark_returns, risk_free, trading_days)
    if b == 0:
        return 0.0
    mu, _ = annualized_return_vol(returns, trading_days)
    return float((mu - risk_free) / b)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free: float = None,
    trading_days: int = 252,
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    if risk_free is None:
        risk_free = fetch_risk_free_rate()
    daily_rf = risk_free / trading_days
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(trading_days)).dropna()


def rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    trading_days: int = 252,
) -> pd.Series:
    """Rolling annualized volatility."""
    return (returns.rolling(window).std() * np.sqrt(trading_days)).dropna()


# ---------------------------------------------------------------------------
# Full analytics report
# ---------------------------------------------------------------------------

def compute_analytics(
    value_series: pd.Series,
    benchmark_ticker: str = "SPY",
    risk_free: float = None,
) -> dict:
    """Compute all analytics for a value time series. Returns a dict of metrics."""
    if value_series.empty or len(value_series) < 2:
        return {}

    returns = value_series.pct_change().dropna()
    if risk_free is None:
        risk_free = fetch_risk_free_rate()

    # Benchmark
    start_str = str(value_series.index[0].date()) if hasattr(value_series.index[0], 'date') else str(value_series.index[0])
    end_str = str(value_series.index[-1].date()) if hasattr(value_series.index[-1], 'date') else str(value_series.index[-1])
    bench_rets_df = fetch_returns([benchmark_ticker], start=start_str, end=end_str)
    bench_returns = bench_rets_df[benchmark_ticker] if benchmark_ticker in bench_rets_df.columns else pd.Series(dtype=float)

    dd_val, dd_peak, dd_trough = max_drawdown(value_series)
    bw_year = best_worst_year(returns)
    bw_month = best_worst_month(returns)
    b, a, r2 = beta_alpha(returns, bench_returns, risk_free)

    return {
        "Total Return": total_return(value_series),
        "CAGR": annualized_cagr(value_series),
        "Best Year": f"{bw_year['best_year']} ({bw_year['best_year_ret']:.1%})",
        "Worst Year": f"{bw_year['worst_year']} ({bw_year['worst_year_ret']:.1%})",
        "Best Month": f"{bw_month['best_month']} ({bw_month['best_month_ret']:.1%})",
        "Worst Month": f"{bw_month['worst_month']} ({bw_month['worst_month_ret']:.1%})",
        "Annualized Volatility": annualized_volatility(returns),
        "Max Drawdown": dd_val,
        "Max DD Period": f"{dd_peak} to {dd_trough}",
        "VaR 95% (Historical)": var_historical(returns, 0.95),
        "VaR 99% (Historical)": var_historical(returns, 0.99),
        "VaR 95% (Parametric)": var_parametric(returns, 0.95),
        "CVaR 95%": cvar(returns, 0.95),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free),
        "Sortino Ratio": sortino_ratio(returns, risk_free),
        "Calmar Ratio": calmar_ratio(value_series),
        "Beta": b,
        "Alpha": a,
        "R-Squared": r2,
        "Treynor Ratio": treynor_ratio(returns, bench_returns, risk_free),
    }
