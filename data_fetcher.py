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
try:
    import yfinance as yf
except ImportError:  # curl_cffi build can fail on some platforms (e.g. Streamlit Cloud)
    yf = None  # type: ignore[assignment]

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

_FMP_BASE = "https://financialmodelingprep.com/stable"
_FMP_DAILY_LIMIT = 250
_FMP_WARN_THRESHOLD = 240


def _get_fmp_api_key() -> Optional[str]:
    """Read FMP API key exclusively from Streamlit secrets.

    Expected secrets.toml layout::

        [api_keys]
        FMP_API_KEY = "your-key-here"
    """
    try:
        key = st.secrets["api_keys"]["FMP_API_KEY"]
        if key:
            return str(key)
    except (KeyError, FileNotFoundError, Exception):
        pass
    return None


def _fmp_rate_check() -> bool:
    """Return True if an FMP request is allowed (under daily limit).

    Increments the counter stored in ``st.session_state["fmp_request_count"]``.
    Logs a warning when approaching the limit and blocks at the limit.
    """
    count = st.session_state.get("fmp_request_count", 0)
    if count >= _FMP_DAILY_LIMIT:
        logger.warning("FMP daily rate limit reached (%s/%s)", count, _FMP_DAILY_LIMIT)
        return False
    if count >= _FMP_WARN_THRESHOLD:
        logger.warning("FMP rate limit approaching (%s/%s)", count, _FMP_DAILY_LIMIT)
    st.session_state["fmp_request_count"] = count + 1
    return True


def _fmp_requests_remaining() -> int:
    """Return how many FMP requests remain today."""
    count = st.session_state.get("fmp_request_count", 0)
    return max(0, _FMP_DAILY_LIMIT - count)


def _parse_fmp_date(val) -> Optional[dt.date]:
    """Parse a date string from an FMP response."""
    if not val:
        return None
    try:
        return dt.date.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fmp_dividends(ticker: str, api_key: str) -> pd.DataFrame:
    """Fetch dividend data from Financial Modeling Prep.

    Returns DataFrame with columns:
        ex_dividend_date, declaration_date, record_date, payment_date,
        dividend_per_share, payment_date_source
    All dates are ``datetime.date`` objects.
    Returns empty DataFrame when FMP is unavailable or has no data.

    Uses the **adjDividend** field (split-adjusted) rather than the raw
    ``dividend`` field so amounts align with yfinance's split-adjusted
    history.
    """
    if _requests is None or not api_key:
        return pd.DataFrame()
    if not _fmp_rate_check():
        return pd.DataFrame()

    url = f"{_FMP_BASE}/dividends"
    try:
        resp = _requests.get(
            url, params={"symbol": ticker, "apikey": api_key}, timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("FMP dividends HTTP %s for %s", resp.status_code, ticker)
            return pd.DataFrame()
        data = resp.json()
        # Handle FMP error responses
        if isinstance(data, dict) and "Error Message" in data:
            logger.warning("FMP error for %s: %s", ticker, data["Error Message"])
            return pd.DataFrame()
        # Stable API returns a flat list; legacy returned {"historical": [...]}
        if isinstance(data, dict) and "historical" in data:
            entries = data["historical"]
        elif isinstance(data, list):
            entries = data
        else:
            return pd.DataFrame()
        if not entries:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    rows = []
    for entry in entries:
        # FMP uses "date" for the ex-dividend date in this endpoint
        ex_date = _parse_fmp_date(entry.get("date"))
        if ex_date is None:
            continue
        # Use adjDividend (split-adjusted) for consistency with yfinance
        adj_div = entry.get("adjDividend")
        if adj_div is None:
            adj_div = entry.get("dividend", 0)
        rows.append({
            "ex_dividend_date": ex_date,
            "declaration_date": _parse_fmp_date(entry.get("declarationDate")),
            "record_date": _parse_fmp_date(entry.get("recordDate")),
            "payment_date": _parse_fmp_date(entry.get("paymentDate")),
            "dividend_per_share": float(adj_div or 0),
            "payment_date_source": "FMP",
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ex_dividend_date").reset_index(drop=True)
    return df


def test_fmp_connection() -> dict:
    """Test FMP API connectivity and key validity.

    Returns a dict with keys:
        connected (bool), status_code (int | None),
        error_message (str), rate_limit_remaining (int).
    """
    result = {
        "connected": False,
        "status_code": None,
        "error_message": "",
        "rate_limit_remaining": _fmp_requests_remaining(),
    }
    api_key = _get_fmp_api_key()
    if not api_key:
        result["error_message"] = "No FMP API key found in .streamlit/secrets.toml"
        return result
    if _requests is None:
        result["error_message"] = "requests library not installed"
        return result
    if not _fmp_rate_check():
        result["error_message"] = "FMP daily rate limit reached"
        result["rate_limit_remaining"] = 0
        return result

    try:
        resp = _requests.get(
            f"{_FMP_BASE}/profile",
            params={"symbol": "SPY", "apikey": api_key},
            timeout=10,
        )
        result["status_code"] = resp.status_code

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                result["error_message"] = data["Error Message"]
            elif isinstance(data, list) and len(data) == 0:
                result["error_message"] = "Empty response — key may be invalid"
            elif isinstance(data, list) and len(data) > 0:
                result["connected"] = True
            else:
                result["connected"] = True
        elif resp.status_code == 403:
            # Include response body for diagnostics
            body = ""
            try:
                body = resp.text[:200]
            except Exception:
                pass
            result["error_message"] = f"FMP key invalid or rate-limited (403). {body}".strip()
        else:
            result["error_message"] = f"FMP returned HTTP {resp.status_code}"

        result["rate_limit_remaining"] = _fmp_requests_remaining()
        return result
    except Exception as exc:
        result["error_message"] = f"Connection error: {exc}"
        result["rate_limit_remaining"] = _fmp_requests_remaining()
        return result


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

    Delegates to the **verified** dividend layer which cross-validates
    across yfinance, FMP, SEC EDGAR, and PIMCO.

    Columns: ex_dividend_date, declaration_date, record_date,
             payment_date, payment_date_source, dividend_per_share
    """
    from dividend_verifier import get_verified_dividend_events

    start_date = dt.date.fromisoformat(start) if start else None
    end_date = dt.date.fromisoformat(end) if end else None

    events = get_verified_dividend_events(ticker, start_date=start_date, end_date=end_date)

    if not events:
        return pd.DataFrame(columns=[
            "ex_dividend_date", "declaration_date", "record_date",
            "payment_date", "payment_date_source", "dividend_per_share",
        ])

    rows = []
    for ev in events:
        rows.append({
            "ex_dividend_date": ev.ex_dividend_date,
            "declaration_date": ev.declaration_date,
            "record_date": ev.record_date,
            "payment_date": ev.payment_date,
            "payment_date_source": ev.payment_date_source,
            "dividend_per_share": ev.distribution_per_share,
        })

    return pd.DataFrame(rows).sort_values("ex_dividend_date").reset_index(drop=True)


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
    """Compute the standardized annualized distribution yield using
    the **verified** dividend data layer.

    Formula: last_single_div * distributions_per_year / current_price

    Returns ``(yield_decimal, distributions_per_year, frequency_label)``.
    """
    from dividend_verifier import get_latest_distribution

    latest = get_latest_distribution(ticker)
    if latest is None or latest.distribution_per_share <= 0:
        # Fallback to raw yfinance if verification layer returns nothing
        divs = fetch_dividend_history(ticker)
        if divs.empty:
            return 0.0, 0, "Unknown"
        freq_label, dpy = detect_distribution_frequency(divs)
        if override_freq > 0:
            dpy = override_freq
        if dpy == 0:
            return 0.0, 0, freq_label
        last_div = float(divs.iloc[-1])
        price = fetch_current_price(ticker)
        if not price or price <= 0:
            return 0.0, dpy, freq_label
        ann_yield = (last_div * dpy) / price
        return ann_yield, dpy, _freq_label(dpy, freq_label)

    dpy = latest.frequency
    if override_freq > 0:
        dpy = override_freq

    freq_label = _freq_label(dpy)
    if dpy == 0:
        return 0.0, 0, freq_label

    price = fetch_current_price(ticker)
    if not price or price <= 0:
        return 0.0, dpy, freq_label

    ann_yield = (latest.distribution_per_share * dpy) / price
    return ann_yield, dpy, freq_label


def _freq_label(dpy: int, fallback: str = "Unknown") -> str:
    """Convert distributions per year to a frequency label."""
    if dpy >= 12:
        return "Monthly"
    elif dpy == 4:
        return "Quarterly"
    elif dpy == 2:
        return "Semi-Annual"
    elif dpy == 1:
        return "Annual"
    return fallback


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

    Uses the verified dividend layer (compute_annualized_yield) when
    possible, falling back to standardized formula and yfinance info fields.
    """
    from dividend_verifier import compute_annualized_yield as verified_yield

    # Try verified computation first
    v_yield = verified_yield(ticker)
    if v_yield is not None and v_yield > 0:
        return v_yield

    # Fallback to standardized computation
    std_yield, dpy, _ = compute_standardized_yield(ticker)
    if std_yield > 0:
        return std_yield

    # Last resort: yfinance info
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
