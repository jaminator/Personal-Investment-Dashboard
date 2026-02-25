# Investment Portfolio Dashboard

Interactive investment portfolio dashboard with Monte Carlo simulation,
multi-sleeve rebalancing, backtesting, and comprehensive risk analytics.

## Features

- **Portfolio Holdings Management** — Add any ticker from yfinance, track unrealized G/L, allocation charts
- **Recurring Contributions & Withdrawals** — Multiple streams, 8 frequency options, projected schedules
- **Monte Carlo Simulation** — GBM and Bootstrap methods, configurable simulations, confidence bands
- **Multi-Sleeve Rebalancing** — Target-weight (Mode A), Kelly signal-based (Mode B), custom formula (Mode C)
- **Historical Backtesting** — Equity curves, drawdown charts, benchmark comparison, trade logs
- **Analytics Dashboard** — 20+ metrics including Sharpe, Sortino, VaR, CVaR, Beta, Alpha
- **Future Value Projection** — Waterfall charts, percentile bands, terminal distribution

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Tech Stack

- **UI**: Streamlit
- **Market Data**: yfinance
- **Charts**: Plotly
- **Analytics**: scipy, statsmodels, numpy, pandas
- **Formula Engine**: asteval (sandboxed eval)
