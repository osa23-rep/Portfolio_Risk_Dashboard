"""
Portfolio Risk & Scenario Analysis Dashboard
Interactive Streamlit dashboard.

Run with:
streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from portfolio import (
    ASSETS, SECTORS, generate_price_data, generate_weights,
    compute_returns, portfolio_return, compute_all_metrics,
    run_scenario_analysis, correlation_matrix, risk_contributions,
    compare_rebalancing,
    create_db, save_assets, save_metrics, save_scenarios
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal styling only
st.markdown("""
<style>
    .stDataFrame {
        font-size: 0.85rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Controls")

    n_assets = st.slider(
        "Number of Assets",
        min_value=10,
        max_value=min(55, len(ASSETS)),
        value=min(55, len(ASSETS)),
        step=5
    )

    seed = st.number_input(
        "Random Seed",
        value=42,
        step=1
    )

    risk_free = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=8.0,
        value=4.0,
        step=0.5
    ) / 100

    st.caption("Risk-free rate is used in the Sharpe Ratio calculation.")

    st.divider()

    save_to_db = st.checkbox(
        "Save results to SQL database",
        value=False
    )

    run_btn = st.button(
        "Regenerate Portfolio",
        use_container_width=True
    )

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data(seed_val, n):
    np.random.seed(seed_val)

    prices = generate_price_data()

    # Randomly select assets from the full universe instead of taking the first n.
    # This avoids bias toward tech-heavy names at the beginning of the ASSETS list.
    assets = np.random.choice(ASSETS, size=n, replace=False).tolist()

    weights = generate_weights(assets)

    return prices, weights, assets

if run_btn:
    st.cache_data.clear()

prices, weights, assets = load_data(int(seed), n_assets)
returns = compute_returns(prices)
port_ret = portfolio_return(weights, returns)
metrics = compute_all_metrics(weights, prices[assets])

# Recalculate Sharpe Ratio using selected risk-free rate
if metrics["annualized_vol"] != 0:
    metrics["sharpe_ratio"] = (
        metrics["annualized_return"] - risk_free
    ) / metrics["annualized_vol"]
else:
    metrics["sharpe_ratio"] = 0

# ─────────────────────────────────────────────
# OPTIONAL SQL SAVE
# ─────────────────────────────────────────────

if save_to_db:
    try:
        create_db()
        save_assets(weights, prices[assets])
        save_metrics(metrics)
        st.sidebar.success("Results saved to SQL database.")
    except Exception as e:
        st.sidebar.warning(f"Database save skipped: {e}")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("# Portfolio Risk & Scenario Analysis Dashboard")

st.caption(
    "Simulated multi-asset portfolio. Portfolio value assumed at $1,000,000. "
    "Metrics include annualized return, volatility, Sharpe ratio, maximum drawdown, and 95% VaR."
)

st.markdown(
    f"""
    This dashboard summarizes portfolio performance, downside risk, scenario P&L,
    asset correlations, and rebalancing results.

    **Portfolio universe:** {n_assets} assets  
    **Simulation period:** 3 years  
    **Selected risk-free rate:** {risk_free:.2%}
    """
)

st.divider()

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
col2.metric("Volatility", f"{metrics['annualized_vol']:.2%}")
col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
col5.metric("VaR 95%", f"{metrics['var_95']:.2%}")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Performance",
    "Scenario Analysis",
    "Correlations & Risk",
    "Rebalancing"
])

# ─────────────────────────────────────────────
# TAB 1: PERFORMANCE
# ─────────────────────────────────────────────

with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("Cumulative Portfolio Return")

        cumulative = (1 + port_ret).cumprod()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cumulative.index, cumulative.values, linewidth=2)
        ax.axhline(1, linewidth=0.8, linestyle="--")

        ax.set_ylabel("Cumulative Return")
        ax.set_xlabel("Date")
        ax.set_title("Portfolio Growth Over Time")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Top 10 Holdings")

        top10 = weights.nlargest(10).reset_index()
        top10.columns = ["Asset", "Weight"]
        top10["Sector"] = top10["Asset"].map(SECTORS)
        top10["Weight"] = top10["Weight"].apply(lambda x: f"{x:.2%}")

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Sector Allocation")

    sector_weights = {}

    for asset, w in weights.items():
        sector = SECTORS.get(asset, "Other")
        sector_weights[sector] = sector_weights.get(sector, 0) + w

    sectors = list(sector_weights.keys())
    vals = list(sector_weights.values())

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    bars = ax2.barh(sectors, vals, height=0.6)

    ax2.set_xlabel("Portfolio Weight")
    ax2.set_title("Allocation by Sector")
    ax2.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax2.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=9
        )

    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "Sector allocation helps identify concentration risk. "
        "A portfolio with too much exposure to one sector may be less diversified."
    )

# ─────────────────────────────────────────────
# TAB 2: SCENARIO ANALYSIS
# ─────────────────────────────────────────────

with tab2:
    st.subheader("Market Shock Scenario Analysis")

    st.markdown(
        """
        This section estimates portfolio-level P&L and ending portfolio value under
        different simulated market conditions.
        """
    )

    scenarios_df = run_scenario_analysis(weights, prices[assets])

    if save_to_db:
        try:
            save_scenarios(scenarios_df)
        except Exception as e:
            st.warning(f"Scenario save skipped: {e}")

    display_cols = [
        "Scenario",
        "Market Shock",
        "Portfolio P&L",
        "Portfolio Value",
        "Return"
    ]

    display_df = scenarios_df[display_cols].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Scenario P&L")

    fig3, ax3 = plt.subplots(figsize=(10, 4))

    ax3.bar(
        scenarios_df["Scenario"],
        scenarios_df["raw_pnl"] / 1000,
        width=0.6
    )

    ax3.axhline(0, linewidth=0.8)
    ax3.set_ylabel("P&L ($000s)")
    ax3.set_xlabel("Scenario")
    ax3.set_title("Portfolio Profit and Loss by Scenario")
    ax3.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=30, ha="right", fontsize=8)

    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Positive values represent estimated gains. Negative values represent estimated losses."
    )

# ─────────────────────────────────────────────
# TAB 3: CORRELATIONS & RISK
# ─────────────────────────────────────────────

with tab3:
    col_x, col_y = st.columns(2)

    with col_x:
        st.subheader("Correlation Matrix")

        top20 = weights.nlargest(min(20, len(weights))).index.tolist()
        corr = correlation_matrix(returns[top20])

        fig4, ax4 = plt.subplots(figsize=(8, 7))

        im = ax4.imshow(
            corr.values,
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax4.set_xticks(range(len(top20)))
        ax4.set_yticks(range(len(top20)))
        ax4.set_xticklabels(top20, rotation=90, fontsize=7)
        ax4.set_yticklabels(top20, fontsize=7)
        ax4.set_title("Correlation of Top Holdings")

        plt.colorbar(im, ax=ax4)

        st.pyplot(fig4)
        plt.close(fig4)

        st.caption(
            "Higher positive correlations may reduce diversification benefits."
        )

    with col_y:
        st.subheader("Risk Contributions")

        rc = risk_contributions(weights, returns[assets])
        top_rc = rc.nlargest(15).reset_index()
        top_rc.columns = ["Asset", "Risk Contribution"]
        top_rc["Sector"] = top_rc["Asset"].map(SECTORS)

        fig5, ax5 = plt.subplots(figsize=(6, 6))

        ax5.barh(
            top_rc["Asset"],
            top_rc["Risk Contribution"],
            height=0.6
        )

        ax5.set_xlabel("Risk Contribution")
        ax5.set_title("Top Risk Contributors")
        ax5.grid(True, axis="x", alpha=0.3)

        st.pyplot(fig5)
        plt.close(fig5)

        st.dataframe(
            top_rc,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Large individual risk contributions may indicate concentration risk."
        )

# ─────────────────────────────────────────────
# TAB 4: REBALANCING
# ─────────────────────────────────────────────

with tab4:
    st.subheader("Rebalancing Strategy Comparison")

    st.markdown(
        """
        This section compares portfolio construction approaches using volatility,
        Sharpe Ratio, maximum drawdown, and estimated volatility reduction.
        """
    )

    rebal_df, vol_reduction = compare_rebalancing(prices[assets], assets)

    st.dataframe(
        rebal_df[
            [
                "Strategy",
                "Ann. Return",
                "Volatility",
                "Sharpe",
                "Max DD",
                "Vol Reduction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"Best rebalancing strategy reduced estimated volatility by "
        f"{vol_reduction:.1%} compared with the original portfolio."
    )

    strategies = rebal_df["Strategy"].tolist()
    vols = rebal_df["raw_vol"].tolist()
    sharpes = [float(s) for s in rebal_df["Sharpe"].tolist()]

    fig6, ax6 = plt.subplots(figsize=(10, 4))

    ax6.bar(strategies, vols, width=0.5)

    ax6.set_ylabel("Annualized Volatility")
    ax6.set_xlabel("Strategy")
    ax6.set_title("Volatility by Strategy")
    ax6.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")

    st.pyplot(fig6)
    plt.close(fig6)

    fig7, ax7 = plt.subplots(figsize=(10, 4))

    ax7.bar(strategies, sharpes, width=0.5)

    ax7.set_ylabel("Sharpe Ratio")
    ax7.set_xlabel("Strategy")
    ax7.set_title("Sharpe Ratio by Strategy")
    ax7.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")

    st.pyplot(fig7)
    plt.close(fig7)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()

st.caption(
    "Built with Python, pandas, NumPy, Matplotlib, SQL, and Streamlit. "
    "Analysis uses simulated price data and is intended for educational portfolio risk modeling."
)
