"""
dividend_verifier.py — Cross-validated dividend data layer.

Fetches dividend per-share amounts and payment dates from two sources
(yfinance and FMP) and cross-validates them before handing verified
data to the rest of the application.

Priority waterfall for per-share amount:
    1. FMP adjDividend (split-adjusted)
    2. yfinance ticker.dividends

Priority waterfall for payment date:
    1. FMP paymentDate (if non-null and after ex_dividend_date)
    2. Estimated offset from ex-dividend date
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import requests as _requests
except ImportError:
    _requests = None

logger = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_DEBUG = False


# ---------------------------------------------------------------------------
# DividendEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class DividendEvent:
    """A single verified dividend/distribution event."""
    ticker: str = ""
    ex_dividend_date: Optional[dt.date] = None
    record_date: Optional[dt.date] = None
    payment_date: Optional[dt.date] = None
    declaration_date: Optional[dt.date] = None
    distribution_per_share: float = 0.0
    frequency: int = 0                        # 12, 4, 2, or 1
    payment_date_source: str = "ESTIMATED"    # "FMP" | "ESTIMATED"
    amount_source: str = "yfinance"           # "FMP" | "yfinance"
    amount_verified: bool = False             # True if both sources agree
    data_quality_warnings: list[str] = field(default_factory=list)

    # Per-source raw values (for diagnostics)
    yfinance_amount: Optional[float] = None
    fmp_amount: Optional[float] = None
    fmp_ex_date: Optional[dt.date] = None
    fmp_payment_date: Optional[dt.date] = None
    fmp_record_date: Optional[dt.date] = None
    fmp_declaration_date: Optional[dt.date] = None


# ---------------------------------------------------------------------------
# Source 1: yfinance
# ---------------------------------------------------------------------------

def _fetch_yfinance_dividends(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> list[dict]:
    """Fetch dividends from yfinance. Returns list of dicts with
    ex_dividend_date and dividend_per_share."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        divs = tk.dividends
        if divs is None or divs.empty:
            return []
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        divs = divs.sort_index()
        if start:
            divs = divs[divs.index >= pd.Timestamp(start)]
        if end:
            divs = divs[divs.index <= pd.Timestamp(end)]
        results = []
        for ts, amount in divs.items():
            ex_d = ts.date() if hasattr(ts, "date") else ts
            results.append({
                "ex_dividend_date": ex_d,
                "dividend_per_share": float(amount),
            })
        return results
    except Exception as e:
        logger.warning("yfinance dividend fetch failed for %s: %s", ticker, e)
        return []


# ---------------------------------------------------------------------------
# Source 2: FMP
# ---------------------------------------------------------------------------

def _get_fmp_api_key() -> Optional[str]:
    """Read FMP API key exclusively from Streamlit secrets.

    Expected secrets.toml layout::

        [api_keys]
        FMP_API_KEY = "your-key-here"
    """
    try:
        key = st.secrets["api_keys"]["FMP_API_KEY"]
        if key:
            return str(key).strip()
    except (KeyError, FileNotFoundError, Exception):
        pass
    return None


def _fmp_rate_check() -> bool:
    """Return True if an FMP request is allowed under the daily limit."""
    try:
        from data_fetcher import _fmp_rate_check as _df_rate_check
        return _df_rate_check()
    except ImportError:
        return True


def _fetch_fmp_dividends(ticker: str) -> list[dict]:
    """Fetch dividend data from FMP. Returns list of dicts with
    ex_dividend_date, payment_date, record_date, declaration_date,
    dividend_per_share.

    Uses the **adjDividend** field (split-adjusted) for consistency
    with yfinance's split-adjusted dividend history.
    """
    if _requests is None:
        return []
    api_key = _get_fmp_api_key()
    if not api_key:
        return []
    if not _fmp_rate_check():
        return []

    url = f"{_FMP_BASE}/historical-price-full/stock_dividend/{ticker}"
    if FMP_DEBUG:
        redacted = api_key[:len(api_key) - 20] + "******" if len(api_key) > 20 else "******"
        print(f"[FMP CALL] URL: {url}?apikey={redacted}")
        print(f"[FMP CALL] Ticker: {ticker}, Function: _fetch_fmp_dividends")
    try:
        resp = _requests.get(
            url, params={"apikey": api_key}, timeout=10,
        )
        if FMP_DEBUG:
            print(f"[FMP RESPONSE] Status code: {resp.status_code}")
            print(f"[FMP RESPONSE] Raw body (first 500 chars): {resp.text[:500]}")
        # Handle rate-limit or error responses
        if resp.status_code == 429 or "Limit Reach" in resp.text:
            logger.warning("FMP rate limit reached for %s", ticker)
            return []
        if resp.status_code != 200:
            logger.warning("FMP dividends HTTP %s for %s", resp.status_code, ticker)
            return []
        data = resp.json()
        # Handle FMP error responses
        if isinstance(data, dict) and "Error Message" in data:
            logger.warning("FMP error for %s: %s", ticker, data["Error Message"])
            return []
        # v3 API returns {"symbol": "...", "historical": [...]}
        if isinstance(data, dict) and "historical" in data:
            entries = data["historical"]
        elif isinstance(data, list):
            entries = data
        else:
            return []
        if not entries:
            return []
        if FMP_DEBUG:
            print(f"[FMP PARSED] Events found: {len(entries)}")
            print(f"[FMP PARSED] First event: {entries[0] if entries else 'EMPTY'}")
    except Exception as e:
        logger.warning("FMP dividend fetch failed for %s: %s", ticker, e)
        return []

    results = []
    for entry in entries:
        ex_date = _parse_date(entry.get("date"))
        if ex_date is None:
            continue
        # Use adjDividend (split-adjusted) for consistency with yfinance
        adj_div = entry.get("adjDividend")
        if adj_div is None:
            adj_div = entry.get("dividend", 0)
        results.append({
            "ex_dividend_date": ex_date,
            "declaration_date": _parse_date(entry.get("declarationDate")),
            "record_date": _parse_date(entry.get("recordDate")),
            "payment_date": _parse_date(entry.get("paymentDate")),
            "dividend_per_share": float(adj_div or 0),
        })
    return sorted(results, key=lambda x: x["ex_dividend_date"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_date(val) -> Optional[dt.date]:
    """Parse a date value from various formats."""
    if val is None:
        return None
    if isinstance(val, dt.date):
        return val
    try:
        return dt.date.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _dates_agree(d1: Optional[dt.date], d2: Optional[dt.date], tolerance_days: int = 2) -> bool:
    """Check if two dates agree within tolerance."""
    if d1 is None or d2 is None:
        return False
    return abs((d1 - d2).days) <= tolerance_days


def _amounts_agree(a1: Optional[float], a2: Optional[float], tolerance: float = 0.001) -> bool:
    """Check if two amounts agree within tolerance."""
    if a1 is None or a2 is None:
        return False
    return abs(a1 - a2) <= tolerance


def _detect_frequency(ex_dates: list[dt.date]) -> int:
    """Detect distribution frequency from a list of ex-dividend dates."""
    if len(ex_dates) < 2:
        return 0
    # Use trailing 12 months
    one_year_ago = dt.date.today() - dt.timedelta(days=365)
    recent = [d for d in sorted(ex_dates) if d >= one_year_ago]
    if len(recent) < 2:
        recent = sorted(ex_dates)[-6:]
    if len(recent) < 2:
        return 0

    gaps = [(recent[i + 1] - recent[i]).days for i in range(len(recent) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]

    if median_gap <= 35:
        return 12
    elif median_gap <= 95:
        return 4
    elif median_gap <= 200:
        return 2
    else:
        return 1


def _estimate_payment_date(ex_date: dt.date, frequency: int) -> dt.date:
    """Estimate payment date from ex-dividend date and frequency."""
    if frequency >= 12:
        offset = 15  # Monthly funds: ex-date + 15 days
    else:
        offset = 20  # Quarterly/other: ex-date + 20 days
    return ex_date + dt.timedelta(days=offset)


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_verified_dividend_events(
    ticker: str,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
) -> list[DividendEvent]:
    """Return cross-validated DividendEvents for the ticker.

    FMP-first waterfall:
      Amount: FMP adjDividend (if non-null, non-zero) > yfinance
      Payment date: FMP paymentDate (if non-null and after ex_date) > ESTIMATED
      Ex-dividend date: Always yfinance (FMP used for cross-validation only)
      Record/declaration date: FMP if available, else None
    """
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

    # --- Fetch from both sources ---
    yf_divs = _fetch_yfinance_dividends(ticker, start=start_str, end=end_str)
    fmp_divs = _fetch_fmp_dividends(ticker)

    # Build FMP lookup by ex-date
    fmp_lookup: dict[dt.date, dict] = {}
    for fd in fmp_divs:
        ex_d = fd["ex_dividend_date"]
        if start_date and ex_d < start_date:
            continue
        if end_date and ex_d > end_date:
            continue
        fmp_lookup[ex_d] = fd

    # --- Build DividendEvents from yfinance as the date index ---
    events: list[DividendEvent] = []
    all_ex_dates = [d["ex_dividend_date"] for d in yf_divs]
    frequency = _detect_frequency(all_ex_dates)

    for yf_div in yf_divs:
        ex_date = yf_div["ex_dividend_date"]
        yf_amount = yf_div["dividend_per_share"]

        event = DividendEvent(
            ticker=ticker,
            ex_dividend_date=ex_date,
            distribution_per_share=yf_amount,
            frequency=frequency,
            amount_source="yfinance",
            yfinance_amount=yf_amount,
        )

        # --- Match FMP data (exact match, then +/-1-2 day tolerance) ---
        fmp_data = fmp_lookup.get(ex_date)
        if fmp_data is None:
            for offset in [1, -1, 2, -2]:
                candidate = ex_date + dt.timedelta(days=offset)
                if candidate in fmp_lookup:
                    fmp_data = fmp_lookup[candidate]
                    break

        if fmp_data:
            event.fmp_amount = fmp_data.get("dividend_per_share")
            event.fmp_ex_date = fmp_data.get("ex_dividend_date")
            event.fmp_payment_date = fmp_data.get("payment_date")
            event.fmp_record_date = fmp_data.get("record_date")
            event.fmp_declaration_date = fmp_data.get("declaration_date")

            # --- Amount: FMP first if non-null and non-zero ---
            if event.fmp_amount is not None and event.fmp_amount > 0:
                event.distribution_per_share = event.fmp_amount
                event.amount_source = "FMP"

            # --- Payment date: FMP first if non-null and after ex_date ---
            if event.fmp_payment_date and event.fmp_payment_date > ex_date:
                event.payment_date = event.fmp_payment_date
                event.payment_date_source = "FMP"

            # --- Record and declaration dates: FMP if available ---
            if event.fmp_record_date:
                event.record_date = event.fmp_record_date
            if event.fmp_declaration_date:
                event.declaration_date = event.fmp_declaration_date

            if FMP_DEBUG:
                print(f"[WATERFALL] {ticker} {ex_date}: "
                      f"fmp_payment={event.fmp_payment_date}, "
                      f"fmp_amount={event.fmp_amount}, "
                      f"selected_source={event.payment_date_source}")

        # --- Fill missing payment date with estimate ---
        if event.payment_date is None:
            event.payment_date = _estimate_payment_date(ex_date, frequency)
            event.payment_date_source = "ESTIMATED"

        # --- Cross-validation ---
        _cross_validate_event(event)

        events.append(event)

    return events


def _cross_validate_event(event: DividendEvent) -> None:
    """Cross-validate amounts and dates across FMP and yfinance.
    Populates amount_verified and data_quality_warnings."""
    warnings = []

    # --- Amount cross-validation ---
    if event.yfinance_amount is not None and event.fmp_amount is not None:
        if _amounts_agree(event.yfinance_amount, event.fmp_amount):
            event.amount_verified = True
        else:
            event.amount_verified = False
            warnings.append(
                f"Amount mismatch: yfinance=${event.yfinance_amount:.4f} vs "
                f"FMP=${event.fmp_amount:.4f} "
                f"(diff=${abs(event.yfinance_amount - event.fmp_amount):.4f})"
            )
    else:
        event.amount_verified = False  # Only 1 source

    # --- Ex-date cross-validation (yfinance vs FMP) ---
    if event.fmp_ex_date and event.ex_dividend_date:
        if not _dates_agree(event.ex_dividend_date, event.fmp_ex_date, tolerance_days=1):
            warnings.append(
                f"Ex-date mismatch: yfinance={event.ex_dividend_date} vs "
                f"FMP={event.fmp_ex_date} (diff={(event.fmp_ex_date - event.ex_dividend_date).days}d)"
            )

    event.data_quality_warnings = warnings


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_latest_distribution(
    ticker: str,
    as_of_date: Optional[dt.date] = None,
) -> Optional[DividendEvent]:
    """Return the most recent DividendEvent with ex_dividend_date on or before as_of_date.

    Used by the rebalancing engine to compute yield_as_of(date).
    """
    if as_of_date is None:
        as_of_date = dt.date.today()

    # Fetch last 2 years of data
    start = as_of_date - dt.timedelta(days=730)
    events = get_verified_dividend_events(ticker, start_date=start, end_date=as_of_date)
    if not events:
        return None

    # Filter to events on or before as_of_date
    valid = [e for e in events if e.ex_dividend_date and e.ex_dividend_date <= as_of_date]
    if not valid:
        return None

    return max(valid, key=lambda e: e.ex_dividend_date)


def compute_annualized_yield(
    ticker: str,
    reference_date: Optional[dt.date] = None,
    price_on_reference_date: Optional[float] = None,
) -> Optional[float]:
    """Compute annualized distribution yield using verified data.

    Formula:
        yield = last_distribution_per_share(as_of=reference_date)
                * distributions_per_year
                / price_on_reference_date

    Returns None if no distribution data is available.
    """
    if reference_date is None:
        reference_date = dt.date.today()

    latest = get_latest_distribution(ticker, as_of_date=reference_date)
    if latest is None or latest.distribution_per_share <= 0:
        return None

    dpy = latest.frequency
    if dpy <= 0:
        return None

    price = price_on_reference_date
    if price is None or price <= 0:
        # Try to fetch current price
        try:
            from data_fetcher import fetch_current_price
            price = fetch_current_price(ticker)
        except Exception:
            pass
    if not price or price <= 0:
        return None

    return (latest.distribution_per_share * dpy) / price


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def generate_diagnostic_report(
    tickers: list[str],
    num_events: int = 8,
) -> pd.DataFrame:
    """Generate a diagnostic report comparing FMP vs yfinance data
    for the most recent N events of each ticker."""
    rows = []
    for ticker in tickers:
        events = get_verified_dividend_events(ticker)
        recent = events[-num_events:] if len(events) > num_events else events

        for event in recent:
            ex_date_agree = "\u2014"
            if event.fmp_ex_date and event.ex_dividend_date:
                ex_date_agree = "\u2713" if _dates_agree(
                    event.ex_dividend_date, event.fmp_ex_date, 1
                ) else "\u26a0"

            amount_agree = "\u2014"
            if event.fmp_amount is not None and event.yfinance_amount is not None:
                amount_agree = "\u2713" if _amounts_agree(
                    event.yfinance_amount, event.fmp_amount
                ) else "\u26a0"

            rows.append({
                "Ticker": ticker,
                "Ex-Dividend Date": str(event.ex_dividend_date) if event.ex_dividend_date else "\u2014",
                "yfinance Amount ($)": f"${event.yfinance_amount:.4f}" if event.yfinance_amount is not None else "\u2014",
                "FMP Amount ($)": f"${event.fmp_amount:.4f}" if event.fmp_amount is not None else "\u2014",
                "Amount Used ($)": f"${event.distribution_per_share:.4f}",
                "Amount Source": event.amount_source,
                "Amount Agreement": amount_agree,
                "FMP Payment Date": str(event.fmp_payment_date) if event.fmp_payment_date else "\u2014",
                "Estimated Payment Date": str(_estimate_payment_date(event.ex_dividend_date, event.frequency)) if event.ex_dividend_date else "\u2014",
                "Payment Date Used": str(event.payment_date) if event.payment_date else "\u2014",
                "Payment Date Source": event.payment_date_source,
                "FMP Record Date": str(event.fmp_record_date) if event.fmp_record_date else "\u2014",
                "FMP Declaration Date": str(event.fmp_declaration_date) if event.fmp_declaration_date else "\u2014",
                "Warnings": "; ".join(event.data_quality_warnings) if event.data_quality_warnings else "\u2014",
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
