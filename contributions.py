"""
contributions.py — Generate contribution and withdrawal schedules.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from portfolio import ContributionStream, Withdrawal, FREQUENCIES


# ---------------------------------------------------------------------------
# Frequency → date generator
# ---------------------------------------------------------------------------

def _generate_dates(
    frequency: str,
    start: dt.date,
    end: dt.date,
) -> list[dt.date]:
    """Generate a list of contribution/event dates between *start* and *end*."""
    dates: list[dt.date] = []
    if start > end:
        return dates

    if frequency == "Daily":
        cur = start
        while cur <= end:
            if cur.weekday() < 5:  # business days only
                dates.append(cur)
            cur += dt.timedelta(days=1)

    elif frequency == "Weekly":
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += dt.timedelta(weeks=1)

    elif frequency == "Bi-Weekly":
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += dt.timedelta(weeks=2)

    elif frequency == "Semi-Monthly":
        cur = dt.date(start.year, start.month, 1)
        while cur <= end:
            d1 = dt.date(cur.year, cur.month, 1)
            d15 = dt.date(cur.year, cur.month, 15)
            if d1 >= start and d1 <= end:
                dates.append(d1)
            if d15 >= start and d15 <= end:
                dates.append(d15)
            if cur.month == 12:
                cur = dt.date(cur.year + 1, 1, 1)
            else:
                cur = dt.date(cur.year, cur.month + 1, 1)

    elif frequency == "Monthly":
        cur = start
        while cur <= end:
            dates.append(cur)
            m = cur.month + 1
            y = cur.year
            if m > 12:
                m = 1
                y += 1
            day = min(cur.day, _days_in_month(y, m))
            cur = dt.date(y, m, day)

    elif frequency == "Quarterly":
        cur = start
        while cur <= end:
            dates.append(cur)
            m = cur.month + 3
            y = cur.year
            while m > 12:
                m -= 12
                y += 1
            day = min(cur.day, _days_in_month(y, m))
            cur = dt.date(y, m, day)

    elif frequency == "Semi-Annually":
        cur = start
        while cur <= end:
            dates.append(cur)
            m = cur.month + 6
            y = cur.year
            while m > 12:
                m -= 12
                y += 1
            day = min(cur.day, _days_in_month(y, m))
            cur = dt.date(y, m, day)

    elif frequency == "Annually":
        cur = start
        while cur <= end:
            dates.append(cur)
            y = cur.year + 1
            day = min(cur.day, _days_in_month(y, cur.month))
            cur = dt.date(y, cur.month, day)

    return dates


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


# ---------------------------------------------------------------------------
# Build contribution schedule DataFrame
# ---------------------------------------------------------------------------

def build_contribution_schedule(
    streams: list[ContributionStream],
    forecast_start: dt.date,
    forecast_end: dt.date,
) -> pd.DataFrame:
    """Return DataFrame: date, amount, stream_id, allocation_mode, target_ticker, target_sleeve."""
    rows = []
    for s in streams:
        s_start = dt.date.fromisoformat(s.start_date) if s.start_date else forecast_start
        s_end = dt.date.fromisoformat(s.end_date) if s.end_date else forecast_end
        s_start = max(s_start, forecast_start)
        s_end = min(s_end, forecast_end)
        dates = _generate_dates(s.frequency, s_start, s_end)
        for d in dates:
            rows.append({
                "date": d,
                "amount": s.amount,
                "stream_id": s.id,
                "allocation_mode": s.allocation_mode,
                "target_ticker": s.target_ticker,
                "target_sleeve": s.target_sleeve,
            })
    if not rows:
        return pd.DataFrame(columns=["date", "amount", "stream_id", "allocation_mode", "target_ticker", "target_sleeve"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Build withdrawal schedule DataFrame
# ---------------------------------------------------------------------------

def build_withdrawal_schedule(
    withdrawals: list[Withdrawal],
    forecast_start: dt.date,
    forecast_end: dt.date,
) -> pd.DataFrame:
    """Return DataFrame: date, amount, is_percentage, label."""
    rows = []
    for w in withdrawals:
        if not w.date:
            continue
        wd = dt.date.fromisoformat(w.date)
        if forecast_start <= wd <= forecast_end:
            rows.append({
                "date": wd,
                "amount": w.amount,
                "is_percentage": w.is_percentage,
                "label": w.label,
            })
    if not rows:
        return pd.DataFrame(columns=["date", "amount", "is_percentage", "label"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Net cashflow on a given date
# ---------------------------------------------------------------------------

def net_cashflow_on_date(
    contrib_schedule: pd.DataFrame,
    withdrawal_schedule: pd.DataFrame,
    date: dt.date,
    portfolio_value: float = 0.0,
) -> float:
    """Return net cashflow (positive = inflow, negative = outflow) on *date*."""
    inflow = 0.0
    if not contrib_schedule.empty:
        mask = contrib_schedule["date"] == date
        inflow = contrib_schedule.loc[mask, "amount"].sum()

    outflow = 0.0
    if not withdrawal_schedule.empty:
        mask = withdrawal_schedule["date"] == date
        for _, row in withdrawal_schedule[mask].iterrows():
            if row["is_percentage"]:
                outflow += portfolio_value * (row["amount"] / 100.0)
            else:
                outflow += row["amount"]

    return inflow - outflow


# ---------------------------------------------------------------------------
# Frequency → annualization factor
# ---------------------------------------------------------------------------

def frequency_to_annual_fraction(frequency: str) -> float:
    """Return the fraction of a year that one period represents."""
    mapping = {
        "Daily": 1 / 252,
        "Weekly": 1 / 52,
        "Bi-Weekly": 1 / 26,
        "Semi-Monthly": 1 / 24,
        "Monthly": 1 / 12,
        "Quarterly": 1 / 4,
        "Semi-Annually": 1 / 2,
        "Annually": 1.0,
    }
    return mapping.get(frequency, 1 / 12)


def rebalance_dates(
    frequency: str,
    start: dt.date,
    end: dt.date,
) -> list[dt.date]:
    """Generate rebalancing dates based on calendar frequency."""
    return _generate_dates(frequency, start, end)
