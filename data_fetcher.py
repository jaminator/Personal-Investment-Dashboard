"""
data_fetcher.py — yfinance data fetching with Streamlit caching.
"""

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_current_price(ticker: str) -> Optional[float]:
    """Return latest closing price for *ticker*, or None on failure."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """Return OHLCV history for *ticker*. Returns empty DataFrame on failure."""
    try:
        tk = yf.Ticker(ticker)
        if start:
            kw = {"start": start, "end": end or dt.date.today().isoformat()}
        else:
            kw = {"period": period}
        hist = tk.history(**kw)
        if hist.empty:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_histories(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """Return DataFrame of adjusted close prices, one column per ticker."""
    frames = {}
    for t in tickers:
        h = fetch_history(t, start=start, end=end, period=period)
        if not h.empty and "Close" in h.columns:
            frames[t] = h["Close"]
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df = df.sort_index().ffill().dropna(how="all")
    return df


# ---------------------------------------------------------------------------
# Returns helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_returns(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """Daily simple returns for each ticker."""
    prices = fetch_multiple_histories(tickers, start=start, end=end, period=period)
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change().dropna()


def annualized_return_vol(returns: pd.Series, trading_days: int = 252):
    """Return (annualized arithmetic mean return, annualized vol)."""
    if returns.empty or len(returns) < 2:
        return 0.0, 0.0
    mu = returns.mean() * trading_days
    sigma = returns.std() * np.sqrt(trading_days)
    return float(mu), float(sigma)


def cagr(returns: pd.Series, trading_days: int = 252) -> float:
    """Compound annual growth rate from daily returns."""
    if returns.empty or len(returns) < 2:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / trading_days
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


# ---------------------------------------------------------------------------
# Covariance / correlation
# ---------------------------------------------------------------------------

def covariance_matrix(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Annualized covariance matrix."""
    if returns.empty:
        return pd.DataFrame()
    return returns.cov() * trading_days


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix."""
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_risk_free_rate() -> float:
    """Fetch 3-month T-bill rate from ^IRX. Returns annualized decimal."""
    try:
        tk = yf.Ticker("^IRX")
        hist = tk.history(period="5d")
        if hist.empty:
            return 0.05
        rate_pct = float(hist["Close"].dropna().iloc[-1])
        return rate_pct / 100.0
    except Exception:
        return 0.05


# ---------------------------------------------------------------------------
# Distribution yield
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_distribution_yield(ticker: str) -> Optional[float]:
    """Fetch trailing twelve-month distribution yield for *ticker*."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        div_yield = info.get("trailingAnnualDividendYield") or info.get("yield")
        if div_yield is not None:
            return float(div_yield)
        divs = tk.dividends
        if divs.empty:
            return None
        one_year_ago = dt.datetime.now() - dt.timedelta(days=365)
        recent = divs[divs.index >= one_year_ago]
        if recent.empty:
            return None
        price = fetch_current_price(ticker)
        if price and price > 0:
            return float(recent.sum() / price)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lookback period helper
# ---------------------------------------------------------------------------

def lookback_start_date(period: str) -> str:
    """Convert user-facing lookback label to start date string."""
    today = dt.date.today()
    mapping = {
        "1Y": dt.timedelta(days=365),
        "3Y": dt.timedelta(days=365 * 3),
        "5Y": dt.timedelta(days=365 * 5),
        "10Y": dt.timedelta(days=365 * 10),
    }
    delta = mapping.get(period)
    if delta:
        return (today - delta).isoformat()
    return "1900-01-01"  # "max"
