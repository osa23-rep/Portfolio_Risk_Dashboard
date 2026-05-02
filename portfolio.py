"""
Portfolio Risk & Scenario Analysis Dashboard
Core module: data generation, risk metrics, scenario analysis, rebalancing, and SQL storage.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime


ASSETS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "BAC", "GS",
    "V", "MA", "UNH", "JNJ", "PFE", "XOM", "CVX", "WMT", "TGT", "HD",
    "DIS", "NFLX", "PYPL", "ADBE", "CRM", "INTC", "AMD", "QCOM", "MU", "AVGO",
    "SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "GLD", "SLV", "TLT", "HYG",
    "BABA", "TSM", "SONY", "SAP", "ASML", "NVO", "TM", "SHELL", "BP", "RIO",
    "VNQ", "O", "AMT", "PLD", "SCHD"
]


SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Consumer", "TSLA": "Auto",
    "NVDA": "Tech", "META": "Tech", "JPM": "Finance", "BAC": "Finance", "GS": "Finance",
    "V": "Finance", "MA": "Finance", "UNH": "Health", "JNJ": "Health", "PFE": "Health",
    "XOM": "Energy", "CVX": "Energy", "WMT": "Consumer", "TGT": "Consumer", "HD": "Consumer",
    "DIS": "Media", "NFLX": "Media", "PYPL": "Finance", "ADBE": "Tech", "CRM": "Tech",
    "INTC": "Tech", "AMD": "Tech", "QCOM": "Tech", "MU": "Tech", "AVGO": "Tech",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "VTI": "ETF", "EFA": "ETF",
    "EEM": "ETF", "GLD": "Commodity", "SLV": "Commodity", "TLT": "Bond", "HYG": "Bond",
    "BABA": "Tech", "TSM": "Tech", "SONY": "Tech", "SAP": "Tech", "ASML": "Tech",
    "NVO": "Health", "TM": "Auto", "SHELL": "Energy", "BP": "Energy", "RIO": "Materials",
    "VNQ": "REIT", "O": "REIT", "AMT": "REIT", "PLD": "REIT", "SCHD": "ETF"
}


SCENARIOS = {
    "Mild Shock (-10%)": -0.10,
    "Moderate Shock (-20%)": -0.20,
    "Severe Shock (-30%)": -0.30,
    "2008 Crisis (-45%)": -0.45,
    "COVID Crash (-34%)": -0.34,
    "Rate Shock": -0.15,
    "Bull Market (+20%)": 0.20,
}


DB_PATH = "portfolio.db"


def generate_price_data(n_days: int = 756) -> pd.DataFrame:
    """
    Generate simulated daily price data for all assets.

    This version avoids the length mismatch error by making sure:
    len(price_path) == len(dates)
    """

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_days,
        freq="B"
    )

    prices = {}

    for asset in ASSETS:
        mu = np.random.uniform(0.0003, 0.0012)
        sigma = np.random.uniform(0.012, 0.032)
        start_price = np.random.uniform(20, 500)

        daily_returns = np.random.normal(
            loc=mu,
            scale=sigma,
            size=n_days
        )

        price_path = start_price * np.cumprod(1 + daily_returns)

        prices[asset] = price_path

    return pd.DataFrame(prices, index=dates)


def generate_weights(assets: list = None) -> pd.Series:
    """
    Generate random portfolio weights that sum to 1.
    """

    if assets is None:
        assets = ASSETS

    weights = np.random.dirichlet(np.ones(len(assets)) * 2)

    return pd.Series(weights, index=assets)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_return(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    return returns[weights.index].dot(weights)


def annualized_volatility(port_returns: pd.Series) -> float:
    return float(port_returns.std() * np.sqrt(252))


def annualized_return(port_returns: pd.Series) -> float:
    return float(port_returns.mean() * 252)


def sharpe_ratio(port_returns: pd.Series, risk_free: float = 0.04) -> float:
    excess_return = annualized_return(port_returns) - risk_free
    volatility = annualized_volatility(port_returns)

    if volatility == 0:
        return 0.0

    return round(excess_return / volatility, 3)


def max_drawdown(port_returns: pd.Series) -> float:
    cumulative = (1 + port_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return float(drawdown.min())


def value_at_risk(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR. Negative value represents estimated downside loss.
    """

    return float(np.percentile(port_returns, (1 - confidence) * 100))


def asset_volatilities(returns: pd.DataFrame) -> pd.Series:
    return (returns.std() * np.sqrt(252)).round(4)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().round(3)


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """
    Estimate contribution of each asset to total portfolio volatility.
    """

    selected_returns = returns[weights.index]
    covariance_matrix = selected_returns.cov() * 252

    portfolio_variance = float(weights @ covariance_matrix @ weights)
    portfolio_volatility = np.sqrt(portfolio_variance)

    if portfolio_volatility == 0:
        return pd.Series(0, index=weights.index)

    marginal_risk = covariance_matrix @ weights
    risk_contribution = (weights * marginal_risk) / portfolio_volatility

    return risk_contribution.round(6)


def compute_all_metrics(weights: pd.Series, prices: pd.DataFrame) -> dict:
    returns = compute_returns(prices)
    port_returns = portfolio_return(weights, returns)

    return {
        "annualized_return": round(annualized_return(port_returns), 4),
        "annualized_vol": round(annualized_volatility(port_returns), 4),
        "sharpe_ratio": sharpe_ratio(port_returns),
        "max_drawdown": round(max_drawdown(port_returns), 4),
        "var_95": round(value_at_risk(port_returns), 4),
        "num_assets": len(weights),
    }


def run_scenario_analysis(weights: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate portfolio P&L under predefined market shock scenarios.
    """

    portfolio_value = 1_000_000
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
                asset_shock = shock

            shocked_returns[asset] = asset_shock

        portfolio_shock = sum(
            weights[asset] * shocked_returns[asset]
            for asset in weights.index
        )

        shocked_value = portfolio_value * (1 + portfolio_shock)
        pnl = shocked_value - portfolio_value

        results.append({
            "Scenario": scenario,
            "Market Shock": f"{shock:.0%}",
            "Portfolio P&L": f"${pnl:,.0f}",
            "Portfolio Value": f"${shocked_value:,.0f}",
            "Return": f"{portfolio_shock:.2%}",
            "raw_pnl": pnl,
            "raw_return": portfolio_shock,
        })

    return pd.DataFrame(results)


def equal_weight(assets: list) -> pd.Series:
    n = len(assets)
    return pd.Series([1 / n] * n, index=assets)


def risk_parity_weights(returns: pd.DataFrame, assets: list) -> pd.Series:
    vols = asset_volatilities(returns[assets])
    inv_vol = 1 / vols

    return (inv_vol / inv_vol.sum()).round(4)


def min_variance_weights(returns: pd.DataFrame, assets: list) -> pd.Series:
    variances = returns[assets].var() * 252
    inv_var = 1 / variances

    return (inv_var / inv_var.sum()).round(4)


def compare_rebalancing(prices: pd.DataFrame, assets: list) -> tuple:
    returns = compute_returns(prices)

    original_weights = generate_weights(assets)
    equal_weights = equal_weight(assets)
    risk_parity = risk_parity_weights(returns, assets)
    min_variance = min_variance_weights(returns, assets)

    strategies = {
        "Original": original_weights,
        "Equal Weight": equal_weights,
        "Risk Parity": risk_parity,
        "Min Variance": min_variance,
    }

    rows = []

    for name, weights in strategies.items():
        metrics = compute_all_metrics(weights, prices)

        rows.append({
            "Strategy": name,
            "Ann. Return": f"{metrics['annualized_return']:.2%}",
            "Volatility": f"{metrics['annualized_vol']:.2%}",
            "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
            "Max DD": f"{metrics['max_drawdown']:.2%}",
            "raw_vol": metrics["annualized_vol"],
        })

    df = pd.DataFrame(rows)

    original_vol = df.loc[df["Strategy"] == "Original", "raw_vol"].values[0]
    best_vol = df["raw_vol"].min()

    if original_vol == 0:
        reduction = 0
    else:
        reduction = (original_vol - best_vol) / original_vol

    df["Vol Reduction"] = df["raw_vol"].apply(
        lambda vol: f"{(original_vol - vol) / original_vol:.1%}"
        if vol < original_vol and original_vol != 0
        else "—"
    )

    return df, reduction


def create_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            sector TEXT,
            weight REAL,
            price REAL,
            updated TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            ann_return REAL,
            volatility REAL,
            sharpe REAL,
            max_dd REAL,
            var_95 REAL,
            num_assets INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            scenario TEXT,
            market_shock TEXT,
            portfolio_pnl TEXT,
            port_return TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_assets(weights: pd.Series, prices: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    latest_prices = prices.iloc[-1]

    for ticker, weight in weights.items():
        conn.execute(
            """
            INSERT INTO assets (ticker, sector, weight, price)
            VALUES (?, ?, ?, ?)
            """,
            (
                ticker,
                SECTORS.get(ticker, "Other"),
                float(weight),
                float(latest_prices.get(ticker, 0)),
            )
        )

    conn.commit()
    conn.close()


def save_metrics(metrics: dict):
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO metrics (
            run_date, ann_return, volatility, sharpe, max_dd, var_95, num_assets
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics["annualized_return"],
            metrics["annualized_vol"],
            metrics["sharpe_ratio"],
            metrics["max_drawdown"],
            metrics["var_95"],
            metrics["num_assets"],
        )
    )

    conn.commit()
    conn.close()


def save_scenarios(scenarios_df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)

    for _, row in scenarios_df.iterrows():
        conn.execute(
            """
            INSERT INTO scenarios (
                run_date, scenario, market_shock, portfolio_pnl, port_return
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                row["Scenario"],
                row["Market Shock"],
                row["Portfolio P&L"],
                row["Return"],
            )
        )

    conn.commit()
    conn.close()


def query_assets_by_sector(sector: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM assets WHERE sector = ?",
        conn,
        params=(sector,)
    )

    conn.close()

    return df


def query_top_assets_by_weight(n: int = 10) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT ticker, sector, weight, price
        FROM assets
        ORDER BY weight DESC
        LIMIT ?
        """,
        conn,
        params=(n,)
    )

    conn.close()

    return df
