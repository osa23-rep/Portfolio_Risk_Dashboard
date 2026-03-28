# Portfolio Risk & Scenario Analysis Dashboard

Interactive dashboard built with Python, SQL, pandas, NumPy, and Streamlit to track 50+ assets and compute real-time risk metrics.

## Features
- Tracks **55 assets** across Tech, Finance, Health, Energy, ETFs, REITs, and more
- Computes **volatility, Sharpe ratio, and maximum drawdown** in real time
- **Scenario analysis** simulating market shocks from -10% to -30%
- **Correlation matrix** and **risk contribution** analysis to identify concentration risk
- **Rebalancing strategies** (Equal Weight, Risk Parity, Min Variance) that reduce volatility by up to 15%
- All data saved to **SQLite database** with SQL queries by sector, weight, and date

## Files
| File | Description |
|------|-------------|
| `portfolio.py` | Core engine — metrics, scenarios, rebalancing, SQL |
| `dashboard.py` | Streamlit interactive dashboard |

## How to Run

### Install dependencies
```bash
pip install streamlit pandas numpy matplotlib
```

### Launch dashboard
```bash
streamlit run dashboard.py
```

### Use portfolio module directly
```python
from portfolio import *

prices  = generate_price_data()
weights = generate_weights()
metrics = compute_all_metrics(weights, prices)
print(metrics)

scenarios = run_scenario_analysis(weights, prices)
print(scenarios)

rebal_df, vol_reduction = compare_rebalancing(prices, ASSETS)
print(f"Volatility reduced by: {vol_reduction:.1%}")
```

## Dashboard Tabs
- **Performance** — cumulative returns, top holdings, sector allocation
- **Scenario Analysis** — P&L under 7 market shock scenarios
- **Correlations & Risk** — heatmap + risk contribution per asset
- **Rebalancing** — strategy comparison with volatility reduction metrics
