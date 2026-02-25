"""
simulation.py — Monte Carlo simulation engine (GBM and Bootstrap).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from contributions import (
    build_contribution_schedule,
    build_withdrawal_schedule,
    net_cashflow_on_date,
)
from data_fetcher import (
    annualized_return_vol,
    cagr,
    covariance_matrix,
    fetch_returns,
    lookback_start_date,
)
from portfolio import PortfolioConfig


# ---------------------------------------------------------------------------
# Derive portfolio return parameters from holdings
# ---------------------------------------------------------------------------

def portfolio_return_params(
    config: PortfolioConfig,
) -> tuple[float, float]:
    """Return (annualized expected return, annualized volatility) for the
    portfolio, either from historical data or manual inputs."""

    if config.return_source == "Manual":
        return config.manual_return, config.manual_volatility

    tickers = config.tickers()
    if not tickers:
        return config.manual_return, config.manual_volatility

    start = lookback_start_date(config.lookback_period)
    rets = fetch_returns(tickers, start=start)
    if rets.empty:
        return config.manual_return, config.manual_volatility

    w = config.weights()
    weight_arr = np.array([w.get(t, 0.0) for t in rets.columns])
    if weight_arr.sum() == 0:
        return config.manual_return, config.manual_volatility
    weight_arr = weight_arr / weight_arr.sum()

    port_rets = rets.values @ weight_arr
    port_series = pd.Series(port_rets, index=rets.index)
    mu, sigma = annualized_return_vol(port_series)
    return mu, sigma


# ---------------------------------------------------------------------------
# Monte Carlo — Geometric Brownian Motion
# ---------------------------------------------------------------------------

def monte_carlo_gbm(
    initial_value: float,
    annual_return: float,
    annual_vol: float,
    days: int,
    n_sims: int,
    contrib_schedule: pd.DataFrame,
    withdrawal_schedule: pd.DataFrame,
    start_date: dt.date,
) -> np.ndarray:
    """Run GBM Monte Carlo. Returns array of shape (n_sims, days+1)."""
    dt_frac = 1 / 252
    drift = (annual_return - 0.5 * annual_vol ** 2) * dt_frac
    diffusion = annual_vol * np.sqrt(dt_frac)

    paths = np.zeros((n_sims, days + 1))
    paths[:, 0] = initial_value

    rng = np.random.default_rng(42)
    shocks = rng.normal(0, 1, (n_sims, days))

    date_list = pd.bdate_range(start=start_date, periods=days + 1).date

    for t in range(1, days + 1):
        growth = np.exp(drift + diffusion * shocks[:, t - 1])
        paths[:, t] = paths[:, t - 1] * growth

        # Apply cashflows
        if t < len(date_list):
            cf = net_cashflow_on_date(
                contrib_schedule, withdrawal_schedule,
                date_list[t], float(np.median(paths[:, t])),
            )
            if cf != 0:
                paths[:, t] += cf

        # Floor at zero (no negative portfolio values)
        paths[:, t] = np.maximum(paths[:, t], 0)

    return paths


# ---------------------------------------------------------------------------
# Monte Carlo — Bootstrap Historical Returns
# ---------------------------------------------------------------------------

def monte_carlo_bootstrap(
    initial_value: float,
    config: PortfolioConfig,
    days: int,
    n_sims: int,
    contrib_schedule: pd.DataFrame,
    withdrawal_schedule: pd.DataFrame,
    start_date: dt.date,
) -> np.ndarray:
    """Run bootstrap Monte Carlo by resampling historical daily returns."""
    tickers = config.tickers()
    if not tickers:
        mu, sigma = config.manual_return, config.manual_volatility
        return monte_carlo_gbm(
            initial_value, mu, sigma, days, n_sims,
            contrib_schedule, withdrawal_schedule, start_date,
        )

    start_str = lookback_start_date(config.lookback_period)
    rets = fetch_returns(tickers, start=start_str)
    if rets.empty or len(rets) < 20:
        mu, sigma = config.manual_return, config.manual_volatility
        return monte_carlo_gbm(
            initial_value, mu, sigma, days, n_sims,
            contrib_schedule, withdrawal_schedule, start_date,
        )

    w = config.weights()
    weight_arr = np.array([w.get(t, 0.0) for t in rets.columns])
    if weight_arr.sum() == 0:
        weight_arr = np.ones(len(rets.columns)) / len(rets.columns)
    else:
        weight_arr = weight_arr / weight_arr.sum()

    port_rets = (rets.values @ weight_arr)
    n_hist = len(port_rets)

    rng = np.random.default_rng(42)
    sample_idx = rng.integers(0, n_hist, size=(n_sims, days))

    paths = np.zeros((n_sims, days + 1))
    paths[:, 0] = initial_value

    date_list = pd.bdate_range(start=start_date, periods=days + 1).date

    for t in range(1, days + 1):
        daily_ret = port_rets[sample_idx[:, t - 1]]
        paths[:, t] = paths[:, t - 1] * (1 + daily_ret)

        if t < len(date_list):
            cf = net_cashflow_on_date(
                contrib_schedule, withdrawal_schedule,
                date_list[t], float(np.median(paths[:, t])),
            )
            if cf != 0:
                paths[:, t] += cf

        paths[:, t] = np.maximum(paths[:, t], 0)

    return paths


# ---------------------------------------------------------------------------
# Run simulation (dispatcher)
# ---------------------------------------------------------------------------

def run_simulation(config: PortfolioConfig) -> dict:
    """Run the full Monte Carlo simulation and return results dict."""
    today = dt.date.today()
    trading_days = int(config.forecast_years * 252)
    end_date = today + dt.timedelta(days=int(config.forecast_years * 365.25))

    contrib_sched = build_contribution_schedule(config.contributions, today, end_date)
    withdrawal_sched = build_withdrawal_schedule(config.withdrawals, today, end_date)

    initial_value = config.total_value
    if initial_value <= 0:
        initial_value = 1.0  # avoid zero-start issues

    mu, sigma = portfolio_return_params(config)

    if config.simulation_method == "GBM":
        paths = monte_carlo_gbm(
            initial_value, mu, sigma, trading_days, config.num_simulations,
            contrib_sched, withdrawal_sched, today,
        )
    else:
        paths = monte_carlo_bootstrap(
            initial_value, config, trading_days, config.num_simulations,
            contrib_sched, withdrawal_sched, today,
        )

    dates = pd.bdate_range(start=today, periods=trading_days + 1).date

    # Percentiles
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    mean = paths.mean(axis=0)

    # Probability of reaching target
    final_values = paths[:, -1]
    prob_target = float((final_values >= config.target_value).mean())

    # Probability of ruin
    ruin_mask = (paths <= 0).any(axis=1)
    prob_ruin = float(ruin_mask.mean())

    # Total contributions and withdrawals
    total_contributions = contrib_sched["amount"].sum() if not contrib_sched.empty else 0.0
    total_withdrawals = 0.0
    if not withdrawal_sched.empty:
        for _, row in withdrawal_sched.iterrows():
            if row["is_percentage"]:
                total_withdrawals += initial_value * (row["amount"] / 100.0)
            else:
                total_withdrawals += row["amount"]

    median_ending = float(p50[-1])
    estimated_returns = median_ending - initial_value - total_contributions + total_withdrawals

    return {
        "dates": list(dates),
        "paths": paths,
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "mean": mean,
        "prob_target": prob_target,
        "prob_ruin": prob_ruin,
        "initial_value": initial_value,
        "total_contributions": total_contributions,
        "total_withdrawals": total_withdrawals,
        "estimated_returns": estimated_returns,
        "median_ending": median_ending,
        "mu": mu,
        "sigma": sigma,
        "contrib_schedule": contrib_sched,
        "withdrawal_schedule": withdrawal_sched,
    }
