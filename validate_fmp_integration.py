"""
validate_fmp_integration.py — End-to-end FMP integration tests.

Tests:
  A. Secrets access — _get_fmp_api_key() reads from st.secrets only
  B. Connection test — test_fmp_connection() returns correct dict shape
  C. adjDividend field — fetch_fmp_dividends uses adjDividend (split-adjusted)
  D. Rate limit counter — _fmp_rate_check increments and blocks at limit

Run:  python validate_fmp_integration.py
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Minimal Streamlit stub for non-Streamlit environments
# ---------------------------------------------------------------------------

class _FakeSecrets:
    """Mimics st.secrets with nested key access."""
    def __init__(self, data: dict | None = None):
        self._data = data or {}

    def __getitem__(self, key):
        val = self._data[key]
        if isinstance(val, dict):
            return _FakeSecrets(val)
        return val


class _FakeSessionState(dict):
    pass


class _FakeCacheData:
    """No-op decorator that transparently calls the wrapped function."""
    def __call__(self, *args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

    def clear(self):
        pass


class _FakeSt:
    secrets = _FakeSecrets()
    session_state = _FakeSessionState()
    cache_data = _FakeCacheData()


# Patch streamlit before importing project modules
sys.modules.setdefault("streamlit", _FakeSt())  # type: ignore[arg-type]

import importlib

# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
SKIP = 0


def log(status: str, msg: str) -> None:
    global PASS, FAIL, SKIP
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    elif status == "SKIP":
        SKIP += 1
    print(f"  [{status}] {msg}")


# ===== Test A: Secrets access ==============================================
def test_a_secrets_access():
    print("\n=== Test A: Secrets access ===")

    import streamlit as st

    # A1 — No key configured → returns None
    st.secrets = _FakeSecrets({})
    st.session_state = _FakeSessionState()
    # Force reimport to pick up patched st
    import data_fetcher
    importlib.reload(data_fetcher)

    key = data_fetcher._get_fmp_api_key()
    if key is None:
        log("PASS", "A1: _get_fmp_api_key() returns None when no secret configured")
    else:
        log("FAIL", f"A1: Expected None, got {key!r}")

    # A2 — Key in session_state should NOT be read
    st.session_state["fmp_api_key"] = "session-key-should-not-be-used"
    importlib.reload(data_fetcher)
    key = data_fetcher._get_fmp_api_key()
    if key is None:
        log("PASS", "A2: _get_fmp_api_key() ignores session_state (secrets-only)")
    else:
        log("FAIL", f"A2: Expected None (ignore session_state), got {key!r}")

    # A3 — Key in secrets → returns it
    st.secrets = _FakeSecrets({"api_keys": {"FMP_API_KEY": "test-key-123"}})
    importlib.reload(data_fetcher)
    key = data_fetcher._get_fmp_api_key()
    if key == "test-key-123":
        log("PASS", "A3: _get_fmp_api_key() reads from st.secrets['api_keys']['FMP_API_KEY']")
    else:
        log("FAIL", f"A3: Expected 'test-key-123', got {key!r}")

    # A4 — Empty string key → returns None
    st.secrets = _FakeSecrets({"api_keys": {"FMP_API_KEY": ""}})
    importlib.reload(data_fetcher)
    key = data_fetcher._get_fmp_api_key()
    if key is None:
        log("PASS", "A4: _get_fmp_api_key() returns None for empty key")
    else:
        log("FAIL", f"A4: Expected None for empty key, got {key!r}")

    # Clean up
    st.session_state.clear()


# ===== Test B: test_fmp_connection() return shape ===========================
def test_b_connection_test():
    print("\n=== Test B: test_fmp_connection() return shape ===")

    import streamlit as st
    st.secrets = _FakeSecrets({})
    st.session_state = _FakeSessionState()

    import data_fetcher
    importlib.reload(data_fetcher)

    result = data_fetcher.test_fmp_connection()
    required_keys = {"connected", "status_code", "error_message", "rate_limit_remaining"}
    if required_keys.issubset(result.keys()):
        log("PASS", f"B1: Return dict has all required keys: {sorted(required_keys)}")
    else:
        log("FAIL", f"B1: Missing keys: {required_keys - result.keys()}")

    if isinstance(result["connected"], bool):
        log("PASS", "B2: 'connected' is a bool")
    else:
        log("FAIL", f"B2: 'connected' should be bool, got {type(result['connected'])}")

    if isinstance(result["rate_limit_remaining"], int):
        log("PASS", "B3: 'rate_limit_remaining' is an int")
    else:
        log("FAIL", f"B3: 'rate_limit_remaining' should be int, got {type(result['rate_limit_remaining'])}")

    # No key → should not be connected
    if not result["connected"]:
        log("PASS", "B4: Not connected when no key configured")
    else:
        log("FAIL", "B4: Should not be connected without key")

    if "No FMP" in result["error_message"]:
        log("PASS", f"B5: Error message indicates no key: {result['error_message']!r}")
    else:
        log("FAIL", f"B5: Expected 'No FMP' in error_message, got {result['error_message']!r}")


# ===== Test C: adjDividend field usage ======================================
def test_c_adj_dividend():
    print("\n=== Test C: adjDividend field usage ===")

    import data_fetcher
    import dividend_verifier

    # Check source code for adjDividend usage
    import inspect
    df_source = inspect.getsource(data_fetcher.fetch_fmp_dividends)
    dv_source = inspect.getsource(dividend_verifier._fetch_fmp_dividends)

    if "adjDividend" in df_source:
        log("PASS", "C1: data_fetcher.fetch_fmp_dividends uses 'adjDividend'")
    else:
        log("FAIL", "C1: data_fetcher.fetch_fmp_dividends does NOT reference 'adjDividend'")

    if "adjDividend" in dv_source:
        log("PASS", "C2: dividend_verifier._fetch_fmp_dividends uses 'adjDividend'")
    else:
        log("FAIL", "C2: dividend_verifier._fetch_fmp_dividends does NOT reference 'adjDividend'")

    # Verify that the old raw "dividend" usage as primary field is gone
    # (it's fine as fallback, but adjDividend should be checked first)
    # Look for the pattern: entry.get("dividend", 0) as the PRIMARY field
    # The new code should try adjDividend first, then fallback to dividend
    if 'entry.get("adjDividend")' in df_source or "entry.get('adjDividend')" in df_source:
        log("PASS", "C3: data_fetcher tries adjDividend first (before fallback)")
    else:
        log("FAIL", "C3: data_fetcher should try adjDividend first")

    if 'entry.get("adjDividend")' in dv_source or "entry.get('adjDividend')" in dv_source:
        log("PASS", "C4: dividend_verifier tries adjDividend first (before fallback)")
    else:
        log("FAIL", "C4: dividend_verifier should try adjDividend first")


# ===== Test D: Rate limit counter ==========================================
def test_d_rate_limit():
    print("\n=== Test D: Rate limit counter ===")

    import streamlit as st
    st.session_state = _FakeSessionState()

    import data_fetcher
    importlib.reload(data_fetcher)

    # D1 — Counter starts at 0
    remaining = data_fetcher._fmp_requests_remaining()
    if remaining == 250:
        log("PASS", f"D1: Starts with {remaining} requests remaining")
    else:
        log("FAIL", f"D1: Expected 250 remaining, got {remaining}")

    # D2 — Rate check increments counter
    result = data_fetcher._fmp_rate_check()
    if result is True:
        log("PASS", "D2: _fmp_rate_check() returns True when under limit")
    else:
        log("FAIL", "D2: _fmp_rate_check() should return True when under limit")

    count = st.session_state.get("fmp_request_count", 0)
    if count == 1:
        log("PASS", f"D3: Counter incremented to {count}")
    else:
        log("FAIL", f"D3: Expected count=1, got {count}")

    # D4 — At limit → blocks
    st.session_state["fmp_request_count"] = 250
    importlib.reload(data_fetcher)
    result = data_fetcher._fmp_rate_check()
    if result is False:
        log("PASS", "D4: _fmp_rate_check() returns False at limit (250)")
    else:
        log("FAIL", "D4: Should return False at limit")

    remaining = data_fetcher._fmp_requests_remaining()
    if remaining == 0:
        log("PASS", f"D5: 0 requests remaining at limit")
    else:
        log("FAIL", f"D5: Expected 0 remaining, got {remaining}")

    # D6 — Warning threshold
    st.session_state["fmp_request_count"] = 240
    importlib.reload(data_fetcher)
    result = data_fetcher._fmp_rate_check()
    if result is True:
        log("PASS", "D6: _fmp_rate_check() still True at warn threshold (240)")
    else:
        log("FAIL", "D6: Should still allow at 240")

    # Clean up
    st.session_state.clear()


# ===== Test E: Secrets-only in dividend_verifier ============================
def test_e_verifier_secrets():
    print("\n=== Test E: dividend_verifier secrets access ===")

    import streamlit as st

    # E1 — No key configured → returns None
    st.secrets = _FakeSecrets({})
    st.session_state = _FakeSessionState()

    import dividend_verifier
    importlib.reload(dividend_verifier)

    key = dividend_verifier._get_fmp_api_key()
    if key is None:
        log("PASS", "E1: dividend_verifier._get_fmp_api_key() returns None without secret")
    else:
        log("FAIL", f"E1: Expected None, got {key!r}")

    # E2 — session_state key should be ignored
    st.session_state["fmp_api_key"] = "should-be-ignored"
    importlib.reload(dividend_verifier)
    key = dividend_verifier._get_fmp_api_key()
    if key is None:
        log("PASS", "E2: dividend_verifier ignores session_state")
    else:
        log("FAIL", f"E2: Expected None, got {key!r}")

    # E3 — Key from secrets
    st.secrets = _FakeSecrets({"api_keys": {"FMP_API_KEY": "verifier-key"}})
    importlib.reload(dividend_verifier)
    key = dividend_verifier._get_fmp_api_key()
    if key == "verifier-key":
        log("PASS", "E3: dividend_verifier reads from st.secrets['api_keys']['FMP_API_KEY']")
    else:
        log("FAIL", f"E3: Expected 'verifier-key', got {key!r}")

    st.session_state.clear()


# ===== Main =================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FMP Integration Validation Tests")
    print("=" * 60)

    test_a_secrets_access()
    test_b_connection_test()
    test_c_adj_dividend()
    test_d_rate_limit()
    test_e_verifier_secrets()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    print("\nAll tests passed!")
