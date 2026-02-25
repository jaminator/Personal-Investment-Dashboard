"""
data_fetcher.py — yfinance data fetching with Streamlit caching,
plus dividend data (yfinance + FMP fallback), frequency detection,
and total-return calculations.
"""

import datetime as dt
import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import requests as _requests
except ImportError:  # graceful if requests not installed
    _requests = None

logger = logging.getLogger(__name__)

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
# Dividend data — yfinance (primary)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Fetch historical dividends from yfinance for *ticker*.

    Returns a Series indexed by **ex-dividend date** with dividend-per-share
    values.  Returns empty Series on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        divs = tk.dividends
        if divs is None or divs.empty:
            return pd.Series(dtype=float)
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        divs = divs.sort_index()
        if start:
            divs = divs[divs.index >= pd.Timestamp(start)]
        if end:
            divs = divs[divs.index <= pd.Timestamp(end)]
        return divs
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multiple_dividend_histories(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.Series]:
    """Fetch dividend history for multiple tickers.

    Returns ``{ticker: pd.Series}`` where each Series is indexed by
    ex-dividend date.
    """
    result: dict[str, pd.Series] = {}
    for t in tickers:
        divs = fetch_dividend_history(t, start=start, end=end)
        if not divs.empty:
            result[t] = divs
    return result


# ---------------------------------------------------------------------------
# Dividend data — FMP fallback
# ---------------------------------------------------------------------------

_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _get_fmp_api_key() -> Optional[str]:
    """Read FMP API key from Streamlit secrets or session state."""
    try:
        key = st.secrets.get("FMP_API_KEY", "")
        if key:
            return str(key)
    except Exception:
        pass
    return st.session_state.get("fmp_api_key", "") or None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fmp_dividends(ticker: str, api_key: str) -> pd.DataFrame:
    """Fetch dividend data from Financial Modeling Prep.

    Returns DataFrame with columns:
        ex_dividend_date, declaration_date, record_date, payment_date,
        dividend_per_share, payment_date_source
    All dates are ``datetime.date`` objects.
    Returns empty DataFrame when FMP is unavailable or has no data.
    """
    if _requests is None or not api_key:
        return pd.DataFrame()
    url = f"{_FMP_BASE}/historical-price-full/stock_dividend/{ticker}"
    try:
        resp = _requests.get(url, params={"apikey": api_key}, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data or "historical" not in data:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    rows = []
    for entry in data["historical"]:
        def _parse_date(val):
            if not val:
                return None
            try:
                return dt.date.fromisoformat(str(val))
            except (ValueError, TypeError):
                return None

        # FMP uses "date" for the ex-dividend date in this endpoint
        ex_date = _parse_date(entry.get("date"))
        if ex_date is None:
            continue
        rows.append({
            "ex_dividend_date": ex_date,
            "declaration_date": _parse_date(entry.get("declarationDate")),
            "record_date": _parse_date(entry.get("recordDate")),
            "payment_date": _parse_date(entry.get("paymentDate")),
            "dividend_per_share": float(entry.get("dividend", 0) or 0),
            "payment_date_source": "FMP",
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ex_dividend_date").reset_index(drop=True)
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def validate_fmp_key(api_key: str) -> str:
    """Check FMP API key validity.  Returns 'valid', 'invalid', or 'error'."""
    if _requests is None or not api_key:
        return "error"
    try:
        resp = _requests.get(
            f"{_FMP_BASE}/historical-price-full/stock_dividend/AAPL",
            params={"apikey": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "historical" in data:
                return "valid"
            # Rate-limit or invalid key often returns error message
            if isinstance(data, dict) and "Error Message" in data:
                return "invalid"
        if resp.status_code == 403:
            return "invalid"
        return "error"
    except Exception:
        return "error"


# ---------------------------------------------------------------------------
# Merged dividend data — yfinance + FMP waterfall
# ---------------------------------------------------------------------------

def get_dividend_events(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    payment_date_offset: int = 20,
) -> pd.DataFrame:
    """Return a comprehensive dividend events table for *ticker*.

    Merges yfinance (primary) with FMP (fallback for payment dates).
    Columns: ex_dividend_date, declaration_date, record_date,
             payment_date, payment_date_source, dividend_per_share
    """
    yf_divs = fetch_dividend_history(ticker, start=start, end=end)
    if yf_divs.empty:
        return pd.DataFrame(columns=[
            "ex_dividend_date", "declaration_date", "record_date",
            "payment_date", "payment_date_source", "dividend_per_share",
        ])

    # Build base rows from yfinance
    rows = []
    for ex_date_ts, div_amount in yf_divs.items():
        ex_date = ex_date_ts.date() if hasattr(ex_date_ts, "date") else ex_date_ts
        rows.append({
            "ex_dividend_date": ex_date,
            "declaration_date": None,
            "record_date": None,
            "payment_date": None,
            "payment_date_source": "Estimated",
            "dividend_per_share": float(div_amount),
        })

    df = pd.DataFrame(rows)

    # Try FMP enrichment
    api_key = _get_fmp_api_key()
    if api_key:
        fmp_df = fetch_fmp_dividends(ticker, api_key)
        if not fmp_df.empty:
            # Build lookup by ex-date
            fmp_lookup = {}
            for _, frow in fmp_df.iterrows():
                fmp_lookup[frow["ex_dividend_date"]] = frow

            for idx, row in df.iterrows():
                ex_d = row["ex_dividend_date"]
                fmp_row = fmp_lookup.get(ex_d)
                if fmp_row is not None:
                    if fmp_row.get("declaration_date"):
                        df.at[idx, "declaration_date"] = fmp_row["declaration_date"]
                    if fmp_row.get("record_date"):
                        df.at[idx, "record_date"] = fmp_row["record_date"]
                    if fmp_row.get("payment_date"):
                        df.at[idx, "payment_date"] = fmp_row["payment_date"]
                        df.at[idx, "payment_date_source"] = "FMP"

    # Fill missing payment dates with offset estimate
    for idx, row in df.iterrows():
        if row["payment_date"] is None:
            ex_d = row["ex_dividend_date"]
            df.at[idx, "payment_date"] = ex_d + dt.timedelta(days=payment_date_offset)
            df.at[idx, "payment_date_source"] = "Estimated"
        # Fill missing record date: ex-date + 1 business day
        if row["record_date"] is None:
            ex_d = row["ex_dividend_date"]
            rd = ex_d + dt.timedelta(days=1)
            while rd.weekday() >= 5:
                rd += dt.timedelta(days=1)
            df.at[idx, "record_date"] = rd

    return df.sort_values("ex_dividend_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution frequency auto-detection
# ---------------------------------------------------------------------------

def detect_distribution_frequency(dividends: pd.Series) -> tuple[str, int]:
    """Auto-detect distribution payment frequency from trailing 12-month
    ex-dividend dates.

    Returns ``(frequency_label, distributions_per_year)``.
    Frequency labels: Monthly, Quarterly, Semi-Annual, Annual, Unknown.
    """
    if dividends.empty or len(dividends) < 2:
        return "Unknown", 0

    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
    recent = dividends[dividends.index >= one_year_ago]
    if len(recent) < 2:
        recent = dividends.tail(6)  # fallback to last few
    if len(recent) < 2:
        return "Unknown", 0

    gaps = recent.index.to_series().diff().dropna().dt.days
    median_gap = float(gaps.median())

    if median_gap <= 35:
        return "Monthly", 12
    elif median_gap <= 95:
        return "Quarterly", 4
    elif median_gap <= 200:
        return "Semi-Annual", 2
    else:
        return "Annual", 1


def compute_standardized_yield(
    ticker: str,
    override_freq: int = 0,
) -> tuple[float, int, str]:
    """Compute the standardized annualized distribution yield.

    Formula: last_single_div * distributions_per_year / current_price

    Returns ``(yield_decimal, distributions_per_year, frequency_label)``.
    """
    divs = fetch_dividend_history(ticker)
    if divs.empty:
        return 0.0, 0, "Unknown"

    freq_label, dpy = detect_distribution_frequency(divs)
    if override_freq > 0:
        dpy = override_freq
        if dpy == 12:
            freq_label = "Monthly"
        elif dpy == 4:
            freq_label = "Quarterly"
        elif dpy == 2:
            freq_label = "Semi-Annual"
        elif dpy == 1:
            freq_label = "Annual"

    if dpy == 0:
        return 0.0, 0, freq_label

    last_div = float(divs.iloc[-1])
    price = fetch_current_price(ticker)
    if not price or price <= 0:
        return 0.0, dpy, freq_label

    ann_yield = (last_div * dpy) / price
    return ann_yield, dpy, freq_label


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_total_returns(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """Daily **total** returns (price change + dividends) for each ticker.

    On each ex-dividend date the dividend per share is added to the
    numerator so that the return reflects income as well as price
    appreciation.
    """
    prices = fetch_multiple_histories(tickers, start=start, end=end, period=period)
    if prices.empty:
        return pd.DataFrame()

    total_rets = prices.pct_change().copy()

    div_histories = fetch_multiple_dividend_histories(tickers, start=start, end=end)
    for t, divs in div_histories.items():
        if t not in prices.columns:
            continue
        prev_close = prices[t].shift(1)
        for ex_ts, div_amt in divs.items():
            # Normalise timestamp
            ex_ts_norm = pd.Timestamp(ex_ts)
            if ex_ts_norm in total_rets.index and ex_ts_norm in prev_close.index:
                pc = prev_close.loc[ex_ts_norm]
                if pc and pc > 0:
                    total_rets.loc[ex_ts_norm, t] += div_amt / pc

    return total_rets.dropna()


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
# Distribution yield (legacy helper — now delegates to standardized)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_distribution_yield(ticker: str) -> Optional[float]:
    """Fetch trailing twelve-month distribution yield for *ticker*.

    Uses the standardized formula (last div * freq / price) when possible,
    falling back to yfinance info fields.
    """
    # Try standardized computation first
    std_yield, dpy, _ = compute_standardized_yield(ticker)
    if std_yield > 0:
        return std_yield

    # Fallback to yfinance info
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
