"""
dividend_verifier.py — Cross-validated dividend data layer.

Fetches dividend per-share amounts and payment dates from multiple
sources (yfinance, FMP, SEC EDGAR, PIMCO) and cross-validates them
before handing verified data to the rest of the application.

Priority waterfall for per-share amount:
    1. SEC EDGAR 8-K filing (most authoritative)
    2. PIMCO distribution page (PDI / PIMCO CEFs only)
    3. FMP historical dividends API
    4. yfinance ticker.dividends

Priority waterfall for payment date:
    1. SEC EDGAR 8-K payable date
    2. PIMCO distribution page payable date (PDI only)
    3. FMP paymentDate field
    4. Estimated offset from ex-dividend date
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import requests as _requests
except ImportError:
    _requests = None

logger = logging.getLogger(__name__)

# SEC EDGAR requires a descriptive User-Agent header
_SEC_HEADERS = {
    "User-Agent": "InvestmentDashboard/1.0 (portfolio-tool@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

_FMP_BASE = "https://financialmodelingprep.com/api/v3"

# Known PIMCO closed-end fund tickers
_PIMCO_CEF_TICKERS = {
    "PDI", "PTY", "PCI", "PKO", "PHK", "PCM", "PFL", "PFN",
    "PDO", "PAXS", "PCN", "PMF", "PGP", "RCS",
}


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
    payment_date_source: str = "ESTIMATED"     # FMP | SEC_EDGAR | ESTIMATED | PIMCO
    amount_source: str = "yfinance"            # yfinance | FMP | SEC_EDGAR | PIMCO
    amount_verified: bool = False              # True if 2+ sources agree
    payment_date_verified: bool = False        # True if 2+ sources agree
    data_quality_warnings: list[str] = field(default_factory=list)

    # Per-source raw values (for diagnostics)
    yfinance_amount: Optional[float] = None
    fmp_amount: Optional[float] = None
    fmp_ex_date: Optional[dt.date] = None
    fmp_payment_date: Optional[dt.date] = None
    fmp_record_date: Optional[dt.date] = None
    fmp_declaration_date: Optional[dt.date] = None
    sec_amount: Optional[float] = None
    sec_payment_date: Optional[dt.date] = None
    pimco_amount: Optional[float] = None
    pimco_payment_date: Optional[dt.date] = None


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
            return str(key)
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
    try:
        resp = _requests.get(url, params={"apikey": api_key}, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Handle FMP error responses
        if isinstance(data, dict) and "Error Message" in data:
            logger.warning("FMP error for %s: %s", ticker, data["Error Message"])
            return []
        if isinstance(data, list) and len(data) == 0:
            return []
        if not data or "historical" not in data:
            return []
    except Exception as e:
        logger.warning("FMP dividend fetch failed for %s: %s", ticker, e)
        return []

    results = []
    for entry in data["historical"]:
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
# Source 3: SEC EDGAR
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sec_cik(ticker: str) -> Optional[str]:
    """Look up the CIK number for a ticker from SEC EDGAR company_tickers.json."""
    if _requests is None:
        return None
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = _requests.get(url, headers=_SEC_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"])
                return cik.zfill(10)  # Pad to 10 digits
        return None
    except Exception as e:
        logger.warning("SEC CIK lookup failed for %s: %s", ticker, e)
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sec_filings(cik: str, form_type: str = "8-K", count: int = 20) -> list[dict]:
    """Fetch recent filings from SEC EDGAR submissions endpoint.

    Returns list of dicts with: accessionNumber, filingDate, primaryDocument, form.
    """
    if _requests is None or not cik:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = _requests.get(url, headers=_SEC_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type:
                results.append({
                    "accessionNumber": accessions[i] if i < len(accessions) else "",
                    "filingDate": dates[i] if i < len(dates) else "",
                    "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
                    "form": form,
                })
                if len(results) >= count:
                    break
        return results
    except Exception as e:
        logger.warning("SEC filings fetch failed for CIK %s: %s", cik, e)
        return []


def _fetch_sec_8k_dividend_info(cik: str, filing: dict) -> Optional[dict]:
    """Fetch and parse an 8-K filing to extract dividend declaration info.

    Returns dict with distribution_per_share and payment_date if found.
    """
    if _requests is None:
        return None

    accession = filing.get("accessionNumber", "").replace("-", "")
    primary_doc = filing.get("primaryDocument", "")
    if not accession or not primary_doc:
        return None

    accession_dashed = filing.get("accessionNumber", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"

    try:
        resp = _requests.get(url, headers=_SEC_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        text = resp.text

        # Look for dividend/distribution per share patterns in the filing
        result = {}

        # Pattern: "$X.XXXX per share" or "distribution of $X.XXXX"
        amount_patterns = [
            r'\$(\d+\.\d{2,6})\s*per\s+(?:common\s+)?share',
            r'distribution\s+(?:of\s+)?\$(\d+\.\d{2,6})',
            r'dividend\s+(?:of\s+)?\$(\d+\.\d{2,6})\s*per',
            r'per\s+share\s+(?:distribution|dividend)\s+(?:of\s+)?\$(\d+\.\d{2,6})',
            r'(\d+\.\d{2,6})\s*per\s+share',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1))
                    if 0.001 < amount < 100.0:  # Sanity check
                        result["distribution_per_share"] = amount
                        break
                except (ValueError, IndexError):
                    continue

        # Pattern: "payable on/date Month DD, YYYY" or "payment date of Month DD, YYYY"
        date_patterns = [
            r'payable\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'payment\s+date\s+(?:of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'pay(?:able)?\s+date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
            r'payable\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'payment\s+date[:\s]+(\d{4}-\d{2}-\d{2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    for fmt in ["%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%Y-%m-%d"]:
                        try:
                            parsed = dt.datetime.strptime(date_str.replace(",", "").strip(), fmt)
                            result["payment_date"] = parsed.date()
                            break
                        except ValueError:
                            continue
                    if "payment_date" in result:
                        break
                except (ValueError, IndexError):
                    continue

        # Pattern: ex-dividend date
        ex_patterns = [
            r'ex-?dividend\s+date\s+(?:of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'ex-?date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
            r'ex-?dividend[:\s]+(\d{4}-\d{2}-\d{2})',
        ]
        for pattern in ex_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    for fmt in ["%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%Y-%m-%d"]:
                        try:
                            parsed = dt.datetime.strptime(date_str.replace(",", "").strip(), fmt)
                            result["ex_dividend_date"] = parsed.date()
                            break
                        except ValueError:
                            continue
                    if "ex_dividend_date" in result:
                        break
                except (ValueError, IndexError):
                    continue

        if result:
            result["filing_date"] = filing.get("filingDate", "")
            return result
        return None

    except Exception as e:
        logger.warning("SEC 8-K parse failed: %s", e)
        return None


def _fetch_sec_dividends(ticker: str, max_filings: int = 10) -> list[dict]:
    """Fetch dividend info from SEC EDGAR 8-K filings.

    Returns list of dicts with distribution_per_share, payment_date,
    ex_dividend_date (when available), filing_date.
    """
    cik = _fetch_sec_cik(ticker)
    if not cik:
        return []

    filings = _fetch_sec_filings(cik, form_type="8-K", count=max_filings)
    if not filings:
        return []

    results = []
    for filing in filings:
        info = _fetch_sec_8k_dividend_info(cik, filing)
        if info and ("distribution_per_share" in info or "payment_date" in info):
            results.append(info)
        if len(results) >= 6:
            break

    return results


# ---------------------------------------------------------------------------
# Source 4: PIMCO distribution page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_pimco_distributions(ticker: str) -> list[dict]:
    """Fetch distribution info from PIMCO's website for CEF tickers.

    Returns list of dicts with ex_dividend_date, payment_date,
    distribution_per_share.
    """
    if _requests is None or ticker.upper() not in _PIMCO_CEF_TICKERS:
        return []

    # PIMCO distribution data can be found at their fund pages
    # The API endpoint pattern for distributions:
    fund_slug_map = {
        "PDI": "pimco-dynamic-income-fund/pdi",
        "PTY": "pimco-corporate-income-opportunity-fund/pty",
        "PCI": "pimco-dynamic-credit-income-fund/pci",
        "PKO": "pimco-income-opportunity-fund/pko",
        "PHK": "pimco-high-income-fund/phk",
        "PCM": "pcm-fund/pcm",
        "PFL": "pimco-income-strategy-fund/pfl",
        "PFN": "pimco-income-strategy-fund-ii/pfn",
        "PDO": "pimco-dynamic-income-opportunities-fund/pdo",
        "PAXS": "pimco-access-income-fund/paxs",
    }

    slug = fund_slug_map.get(ticker.upper())
    if not slug:
        return []

    try:
        url = f"https://www.pimco.com/en-us/investments/closed-end-funds/{slug}"
        resp = _requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; InvestmentDashboard/1.0)",
        }, timeout=15)
        if resp.status_code != 200:
            return []

        text = resp.text
        results = []

        # Look for distribution table data in the HTML
        # PIMCO pages typically list distributions in a table format
        # Pattern: "Ex-Date: MM/DD/YYYY ... Per Share: $X.XXXX ... Payable: MM/DD/YYYY"
        # Or table rows with dates and amounts

        # Try to find per-share amounts and dates
        amount_matches = re.findall(
            r'\$(\d+\.\d{3,6})',
            text,
        )
        date_matches = re.findall(
            r'(\d{1,2}/\d{1,2}/\d{4})',
            text,
        )

        # Parse the structured distribution data if available
        dist_pattern = re.findall(
            r'(?:ex[- ]?(?:date|dividend)[:\s]*)(\d{1,2}/\d{1,2}/\d{4})'
            r'.*?(?:pay(?:able)?[- ]?date[:\s]*)(\d{1,2}/\d{1,2}/\d{4})'
            r'.*?\$(\d+\.\d{3,6})',
            text,
            re.IGNORECASE | re.DOTALL,
        )

        for ex_str, pay_str, amt_str in dist_pattern:
            try:
                ex_date = dt.datetime.strptime(ex_str, "%m/%d/%Y").date()
                pay_date = dt.datetime.strptime(pay_str, "%m/%d/%Y").date()
                amount = float(amt_str)
                results.append({
                    "ex_dividend_date": ex_date,
                    "payment_date": pay_date,
                    "distribution_per_share": amount,
                })
            except (ValueError, IndexError):
                continue

        return results

    except Exception as e:
        logger.warning("PIMCO distribution fetch failed for %s: %s", ticker, e)
        return []


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
    """Return cross-validated DividendEvents for the ticker over the date range.

    Per-share amount priority:
        1. SEC EDGAR 8-K (most authoritative)
        2. PIMCO distribution page (PDI / PIMCO CEFs only)
        3. FMP historical dividends API
        4. yfinance ticker.dividends
    If sources disagree by > $0.001: use highest-confidence source and
    log a data quality warning.

    Payment date priority:
        1. SEC EDGAR 8-K payable date
        2. PIMCO distribution page payable date (PDI only)
        3. FMP paymentDate field
        4. Estimated offset from ex-dividend date
    If sources disagree by > 2 calendar days: use highest-confidence
    source and log a data quality warning.
    """
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

    # --- Fetch from all sources ---
    yf_divs = _fetch_yfinance_dividends(ticker, start=start_str, end=end_str)
    fmp_divs = _fetch_fmp_dividends(ticker)
    sec_divs = _fetch_sec_dividends(ticker, max_filings=10)
    pimco_divs = _fetch_pimco_distributions(ticker)

    # Build FMP lookup by ex-date
    fmp_lookup: dict[dt.date, dict] = {}
    for fd in fmp_divs:
        ex_d = fd["ex_dividend_date"]
        if start_date and ex_d < start_date:
            continue
        if end_date and ex_d > end_date:
            continue
        fmp_lookup[ex_d] = fd

    # Build SEC lookup by ex-date (if available) or filing date proximity
    sec_by_ex: dict[dt.date, dict] = {}
    sec_by_pay: dict[dt.date, dict] = {}
    for sd in sec_divs:
        if "ex_dividend_date" in sd and sd["ex_dividend_date"]:
            sec_by_ex[sd["ex_dividend_date"]] = sd
        if "payment_date" in sd and sd["payment_date"]:
            sec_by_pay[sd["payment_date"]] = sd

    # Build PIMCO lookup by ex-date
    pimco_lookup: dict[dt.date, dict] = {}
    for pd_entry in pimco_divs:
        if "ex_dividend_date" in pd_entry and pd_entry["ex_dividend_date"]:
            pimco_lookup[pd_entry["ex_dividend_date"]] = pd_entry

    # --- Build DividendEvents from yfinance as the base ---
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

        # --- Match FMP data ---
        fmp_data = fmp_lookup.get(ex_date)
        # Try 1-day tolerance if exact match fails
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

            # Use FMP payment date
            if event.fmp_payment_date:
                event.payment_date = event.fmp_payment_date
                event.payment_date_source = "FMP"

            # Use FMP supplementary dates
            if event.fmp_record_date:
                event.record_date = event.fmp_record_date
            if event.fmp_declaration_date:
                event.declaration_date = event.fmp_declaration_date

        # --- Match SEC EDGAR data ---
        sec_data = sec_by_ex.get(ex_date)
        # Try proximity match for SEC data
        if sec_data is None:
            for offset in range(-5, 6):
                candidate = ex_date + dt.timedelta(days=offset)
                if candidate in sec_by_ex:
                    sec_data = sec_by_ex[candidate]
                    break

        if sec_data:
            if "distribution_per_share" in sec_data:
                event.sec_amount = sec_data["distribution_per_share"]
            if "payment_date" in sec_data:
                event.sec_payment_date = sec_data["payment_date"]

            # SEC is highest authority for amounts
            if event.sec_amount is not None and event.sec_amount > 0:
                event.distribution_per_share = event.sec_amount
                event.amount_source = "SEC_EDGAR"

            # SEC is highest authority for payment dates
            if event.sec_payment_date:
                event.payment_date = event.sec_payment_date
                event.payment_date_source = "SEC_EDGAR"

        # --- Match PIMCO data ---
        pimco_data = pimco_lookup.get(ex_date)
        if pimco_data is None and ticker.upper() in _PIMCO_CEF_TICKERS:
            for offset in range(-3, 4):
                candidate = ex_date + dt.timedelta(days=offset)
                if candidate in pimco_lookup:
                    pimco_data = pimco_lookup[candidate]
                    break

        if pimco_data:
            if "distribution_per_share" in pimco_data:
                event.pimco_amount = pimco_data["distribution_per_share"]
            if "payment_date" in pimco_data:
                event.pimco_payment_date = pimco_data["payment_date"]

            # PIMCO overrides FMP/yfinance for amounts (but not SEC)
            if event.pimco_amount and event.amount_source != "SEC_EDGAR":
                event.distribution_per_share = event.pimco_amount
                event.amount_source = "PIMCO"

            # PIMCO overrides FMP for payment dates (but not SEC)
            if event.pimco_payment_date and event.payment_date_source != "SEC_EDGAR":
                event.payment_date = event.pimco_payment_date
                event.payment_date_source = "PIMCO"

        # --- Fill missing payment date with estimate ---
        if event.payment_date is None:
            event.payment_date = _estimate_payment_date(ex_date, frequency)
            event.payment_date_source = "ESTIMATED"

        # --- Fill missing record date ---
        if event.record_date is None:
            rd = ex_date + dt.timedelta(days=1)
            while rd.weekday() >= 5:
                rd += dt.timedelta(days=1)
            event.record_date = rd

        # --- Cross-validation ---
        _cross_validate_event(event)

        events.append(event)

    return events


def _cross_validate_event(event: DividendEvent) -> None:
    """Cross-validate amounts and dates across sources. Populates
    amount_verified, payment_date_verified, and data_quality_warnings."""
    warnings = []

    # --- Amount cross-validation ---
    amounts = {}
    if event.yfinance_amount is not None:
        amounts["yfinance"] = event.yfinance_amount
    if event.fmp_amount is not None:
        amounts["FMP"] = event.fmp_amount
    if event.sec_amount is not None:
        amounts["SEC_EDGAR"] = event.sec_amount
    if event.pimco_amount is not None:
        amounts["PIMCO"] = event.pimco_amount

    if len(amounts) >= 2:
        vals = list(amounts.values())
        all_agree = all(abs(v - vals[0]) <= 0.001 for v in vals)
        if all_agree:
            event.amount_verified = True
        else:
            event.amount_verified = False
            # Find disagreements
            for src1, v1 in amounts.items():
                for src2, v2 in amounts.items():
                    if src1 >= src2:
                        continue
                    if abs(v1 - v2) > 0.001:
                        warnings.append(
                            f"Amount mismatch: {src1}=${v1:.4f} vs {src2}=${v2:.4f} "
                            f"(diff=${abs(v1 - v2):.4f})"
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

    # --- Payment date cross-validation ---
    pay_dates = {}
    if event.fmp_payment_date:
        pay_dates["FMP"] = event.fmp_payment_date
    if event.sec_payment_date:
        pay_dates["SEC_EDGAR"] = event.sec_payment_date
    if event.pimco_payment_date:
        pay_dates["PIMCO"] = event.pimco_payment_date

    if len(pay_dates) >= 2:
        dates_list = list(pay_dates.values())
        all_agree = all(_dates_agree(d, dates_list[0], tolerance_days=2) for d in dates_list)
        if all_agree:
            event.payment_date_verified = True
        else:
            event.payment_date_verified = False
            for src1, d1 in pay_dates.items():
                for src2, d2 in pay_dates.items():
                    if src1 >= src2:
                        continue
                    if not _dates_agree(d1, d2, tolerance_days=2):
                        warnings.append(
                            f"Payment date mismatch: {src1}={d1} vs {src2}={d2} "
                            f"(diff={(d2 - d1).days}d)"
                        )
    elif len(pay_dates) == 1:
        event.payment_date_verified = False  # Only 1 source
    else:
        event.payment_date_verified = False

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
    """Generate a diagnostic report comparing raw data from all sources
    for the most recent N events of each ticker.

    Returns a DataFrame with per-event comparison columns.
    """
    rows = []
    for ticker in tickers:
        events = get_verified_dividend_events(ticker)
        # Take most recent N events
        recent = events[-num_events:] if len(events) > num_events else events

        for event in recent:
            ex_date_agree = "—"
            if event.fmp_ex_date and event.ex_dividend_date:
                ex_date_agree = "Y" if _dates_agree(
                    event.ex_dividend_date, event.fmp_ex_date, 1
                ) else "N"

            amount_agree = "—"
            if event.fmp_amount is not None and event.yfinance_amount is not None:
                amount_agree = "Y" if _amounts_agree(
                    event.yfinance_amount, event.fmp_amount
                ) else "N"

            rows.append({
                "Ticker": ticker,
                "yfinance Ex-Date": str(event.ex_dividend_date) if event.ex_dividend_date else "—",
                "yfinance Dist/Share ($)": f"${event.yfinance_amount:.4f}" if event.yfinance_amount else "—",
                "FMP Ex-Date": str(event.fmp_ex_date) if event.fmp_ex_date else "—",
                "FMP Dist/Share ($)": f"${event.fmp_amount:.4f}" if event.fmp_amount is not None else "—",
                "FMP Payment Date": str(event.fmp_payment_date) if event.fmp_payment_date else "—",
                "FMP Record Date": str(event.fmp_record_date) if event.fmp_record_date else "—",
                "FMP Declaration Date": str(event.fmp_declaration_date) if event.fmp_declaration_date else "—",
                "Date Source Agreement": ex_date_agree,
                "Amount Source Agreement": amount_agree,
                "SEC Amount ($)": f"${event.sec_amount:.4f}" if event.sec_amount is not None else "—",
                "SEC Payment Date": str(event.sec_payment_date) if event.sec_payment_date else "—",
                "PIMCO Amount ($)": f"${event.pimco_amount:.4f}" if event.pimco_amount is not None else "—",
                "PIMCO Payment Date": str(event.pimco_payment_date) if event.pimco_payment_date else "—",
                "Final Amount ($)": f"${event.distribution_per_share:.4f}",
                "Final Amount Source": event.amount_source,
                "Amount Verified": "Y" if event.amount_verified else "N",
                "Final Payment Date": str(event.payment_date) if event.payment_date else "—",
                "Payment Date Source": event.payment_date_source,
                "Pay Date Verified": "Y" if event.payment_date_verified else "N",
                "Warnings": "; ".join(event.data_quality_warnings) if event.data_quality_warnings else "—",
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
