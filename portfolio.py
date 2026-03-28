"""
Portfolio Risk & Scenario Analysis Dashboard
Core module: data generation, risk metrics, scenario analysis, rebalancing.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

ASSETS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM","BAC","GS",
    "V","MA","UNH","JNJ","PFE","XOM","CVX","WMT","TGT","HD",
    "DIS","NFLX","PYPL","ADBE","CRM","INTC","AMD","QCOM","MU","AVGO",
    "SPY","QQQ","IWM","VTI","EFA","EEM","GLD","SLV","TLT","HYG",
    "BABA","TSM","SONY","SAP","ASML","NVO","TM","SHELL","BP","RIO",
    "VNQ","O","AMT","PLD","SCHD"
]

SECTORS = {
    "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","AMZN":"Consumer","TSLA":"Auto",
    "NVDA":"Tech","META":"Tech","JPM":"Finance","BAC":"Finance","GS":"Finance",
    "V":"Finance","MA":"Finance","UNH":"Health","JNJ":"Health","PFE":"Health",
    "XOM":"Energy","CVX":"Energy","WMT":"Consumer","TGT":"Consumer","HD":"Consumer",
    "DIS":"Media","NFLX":"Media","PYPL":"Finance","ADBE":"Tech","CRM":"Tech",
    "INTC":"Tech","AMD":"Tech","QCOM":"Tech","MU":"Tech","AVGO":"Tech",
    "SPY":"ETF","QQQ":"ETF","IWM":"ETF","VTI":"ETF","EFA":"ETF",
    "EEM":"ETF","GLD":"Commodity","SLV":"Commodity","TLT":"Bond","HYG":"Bond",
    "BABA":"Tech","TSM":"Tech","SONY":"Tech","SAP":"Tech","ASML":"Tech",
    "NVO":"Health","TM":"Auto","SHELL":"Energy","BP":"Energy","RIO":"Materials",
    "VNQ":"REIT","O":"REIT","AMT":"REIT","PLD":"REIT","SCHD":"ETF"
}

np.random.seed(42)

def generate_price_data(n_days: int = 756) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq='B')
    prices = {}
    for asset in ASSETS:
        mu    = np.random.uniform(0.0003, 0.0012)
        sigma = np.random.uniform(0.012, 0.032)
        returns = np.random.normal(mu, sigma, n_days)
        start_price = np.random.uniform(20, 500)
        prices[asset] = start_price * np.cumprod(1 + returns)
    return pd.DataFrame(prices, index=dates)

def generate_weights(assets: list = None) -> pd.Series:
    if assets is None:
        assets = ASSETS
    w = np.random.dirichlet(np.ones(len(assets)) * 2)
    return pd.Series(w, index=assets)

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def portfolio_return(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    return returns[weights.index].dot(weights)

def annualized_volatility(port_returns: pd.Series) -> float:
    return float(port_returns.std() * np.sqrt(252))

def annualized_return(port_returns: pd.Series) -> float:
    return float(port_returns.mean() * 252)

def sharpe_ratio(port_returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = annualized_return(port_returns) - risk_free
    vol    = annualized_volatility(port_returns)
    return round(excess / vol, 3) if vol > 0 else 0.0

def max_drawdown(port_returns: pd.Series) -> float:
    cumulative = (1 + port_returns).cumprod()
    roll_max   = cumulative.cummax()
    drawdown   = (cumulative - roll_max) / roll_max
    return float(drawdown.min())

def value_at_risk(port_returns: pd.Series, confidence: float = 0.95) -> float:
    return float(np.percentile(port_returns, (1 - confidence) * 100))

def asset_volatilities(returns: pd.DataFrame) -> pd.Series:
    return (returns.std() * np.sqrt(252)).round(4)

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().round(3)

def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    cov      = returns[weights.index].cov() * 252
    port_vol = float(np.sqrt(weights @ cov @ weights))
    marginal = cov @ weights
    rc = (weights * marginal) / port_vol if port_vol > 0 else weights * 0
    return rc.round(6)

def compute_all_metrics(weights: pd.Series, prices: pd.DataFrame) -> dict:
    returns  = compute_returns(prices)
    port_ret = portfolio_return(weights, returns)
    return {
        "annualized_return": round(annualized_return(port_ret), 4),
        "annualized_vol":    round(annualized_volatility(port_ret), 4),
        "sharpe_ratio":      sharpe_ratio(port_ret),
        "max_drawdown":      round(max_drawdown(port_ret), 4),
        "var_95":            round(value_at_risk(port_ret), 4),
        "num_assets":        len(weights),
    }

SCENARIOS = {
    "Mild Shock (-10%)":     -0.10,
    "Moderate Shock (-20%)": -0.20,
    "Severe Shock (-30%)":   -0.30,
    "2008 Crisis (-45%)":    -0.45,
    "COVID Crash (-34%)":    -0.34,
    "Rate Hike (+10%)":      -0.15,
    "Bull Market (+20%)":    +0.20,
}

def run_scenario_analysis(weights: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    port_value = 1_000_000
    results = []
    for scenario, shock in SCENARIOS.items():
        shocked_returns = {}
        for asset in weights.index:
            sector = SECTORS.get(asset, "Other")
            if sector in ["Tech", "Auto"]:
                asset_shock = shock * 1.3
            elif sector in ["Bond", "Commodity"]:
                asset_shock = shock * 0.3
            elif sector == "Energy":
                asset_shock = shock * 1.1
            else:
                asset_shock = shock * 1.0
            shocked_returns[asset] = asset_shock
        port_shock    = sum(weights[a] * shocked_returns[a] for a in weights.index)
        shocked_value = port_value * (1 + port_shock)
        pnl           = shocked_value - port_value
        results.append({
            "Scenario":        scenario,
            "Market Shock":    f"{shock:.0%}",
            "Portfolio P&L":   f"${pnl:,.0f}",
            "Portfolio Value": f"${shocked_value:,.0f}",
            "Return":          f"{port_shock:.2%}",
            "raw_pnl":         pnl,
            "raw_return":      port_shock,
        })
    return pd.DataFrame(results)

def equal_weight(assets: list) -> pd.Series:
    n = len(assets)
    return pd.Series([1/n]*n, index=assets)

def risk_parity_weights(returns: pd.DataFrame, assets: list) -> pd.Series:
    vols    = asset_volatilities(returns[assets])
    inv_vol = 1 / vols
    return (inv_vol / inv_vol.sum()).round(4)

def min_variance_weights(returns: pd.DataFrame, assets: list) -> pd.Series:
    variances = returns[assets].var() * 252
    inv_var   = 1 / variances
    return (inv_var / inv_var.sum()).round(4)

def compare_rebalancing(prices: pd.DataFrame, assets: list) -> tuple:
    returns = compute_returns(prices)
    orig_w  = generate_weights(assets)
    eq_w    = equal_weight(assets)
    rp_w    = risk_parity_weights(returns, assets)
    mv_w    = min_variance_weights(returns, assets)
    strategies = {"Original": orig_w, "Equal Weight": eq_w, "Risk Parity": rp_w, "Min Variance": mv_w}
    rows = []
    for name, w in strategies.items():
        m = compute_all_metrics(w, prices)
        rows.append({
            "Strategy":    name,
            "Ann. Return": f"{m['annualized_return']:.2%}",
            "Volatility":  f"{m['annualized_vol']:.2%}",
            "Sharpe":      f"{m['sharpe_ratio']:.2f}",
            "Max DD":      f"{m['max_drawdown']:.2%}",
            "raw_vol":     m['annualized_vol'],
        })
    df       = pd.DataFrame(rows)
    orig_vol = df[df['Strategy'] == 'Original']['raw_vol'].values[0]
    best_vol = df['raw_vol'].min()
    reduction = (orig_vol - best_vol) / orig_vol
    df['Vol Reduction'] = df['raw_vol'].apply(
        lambda v: f"{(orig_vol - v)/orig_vol:.1%}" if v < orig_vol else "—"
    )
    return df, reduction

DB_PATH = "portfolio.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, sector TEXT,
        weight REAL, price REAL, updated TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT,
        ann_return REAL, volatility REAL, sharpe REAL, max_dd REAL, var_95 REAL, num_assets INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT,
        scenario TEXT, market_shock TEXT, portfolio_pnl TEXT, port_return TEXT)""")
    conn.commit()
    conn.close()

def save_assets(weights: pd.Series, prices: pd.DataFrame):
    conn   = sqlite3.connect(DB_PATH)
    latest = prices.iloc[-1]
    for ticker, w in weights.items():
        conn.execute("INSERT INTO assets (ticker, sector, weight, price) VALUES (?, ?, ?, ?)",
            (ticker, SECTORS.get(ticker, "Other"), float(w), float(latest.get(ticker, 0))))
    conn.commit()
    conn.close()

def save_metrics(metrics: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO metrics (run_date, ann_return, volatility, sharpe, max_dd, var_95, num_assets) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), metrics['annualized_return'],
         metrics['annualized_vol'], metrics['sharpe_ratio'], metrics['max_drawdown'],
         metrics['var_95'], metrics['num_assets']))
    conn.commit()
    conn.close()

def save_scenarios(scenarios_df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    for _, row in scenarios_df.iterrows():
        conn.execute("INSERT INTO scenarios (run_date, scenario, market_shock, portfolio_pnl, port_return) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), row['Scenario'],
             row['Market Shock'], row['Portfolio P&L'], row['Return']))
    conn.commit()
    conn.close()

def query_assets_by_sector(sector: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM assets WHERE sector = ?", conn, params=(sector,))
    conn.close()
    return df

def query_top_assets_by_weight(n: int = 10) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT ticker, sector, weight, price FROM assets ORDER BY weight DESC LIMIT ?", conn, params=(n,))
    conn.close()
    return df
# Track 55 assets across sectors
# Volatility, Sharpe ratio, max drawdown metrics
