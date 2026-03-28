"""
Portfolio Risk & Scenario Analysis Dashboard
Interactive Streamlit dashboard.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from portfolio import (
    ASSETS, SECTORS, generate_price_data, generate_weights,
    compute_returns, portfolio_return, compute_all_metrics,
    run_scenario_analysis, correlation_matrix, risk_contributions,
    compare_rebalancing, asset_volatilities,
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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
    .metric-card {
        background: #0f1117;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { color: #f0f4ff; font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
    .positive { color: #34d399 !important; }
    .negative { color: #f87171 !important; }
    .stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    n_assets = st.slider("Number of Assets", min_value=10, max_value=55, value=55, step=5)
    seed     = st.number_input("Random Seed", value=42, step=1)
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 8.0, 4.0, 0.5) / 100
    st.divider()
    st.markdown("**Scenario Shock Range**")
    shock_min = st.slider("Min Shock (%)", -50, -5, -30)
    shock_max = st.slider("Max Shock (%)", -5, 30, 20)
    st.divider()
    save_to_db = st.checkbox("Save results to SQL DB", value=True)
    run_btn    = st.button("🔄 Regenerate Portfolio", use_container_width=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data(seed_val, n):
    np.random.seed(seed_val)
    prices  = generate_price_data()
    assets  = ASSETS[:n]
    weights = generate_weights(assets)
    return prices, weights, assets

if run_btn:
    st.cache_data.clear()

prices, weights, assets = load_data(int(seed), n_assets)
returns  = compute_returns(prices)
port_ret = portfolio_return(weights, returns)
metrics  = compute_all_metrics(weights, prices[assets])

# Save to DB
if save_to_db:
    create_db()
    save_assets(weights, prices[assets])
    save_metrics(metrics)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("# 📊 Portfolio Risk & Scenario Analysis")
st.markdown(f"*Tracking **{n_assets} assets** across sectors · 3-year simulation · $1M portfolio*")
st.divider()

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📈 Ann. Return",   f"{metrics['annualized_return']:.2%}")
col2.metric("📉 Volatility",    f"{metrics['annualized_vol']:.2%}")
col3.metric("⚡ Sharpe Ratio",  f"{metrics['sharpe_ratio']:.2f}")
col4.metric("🔻 Max Drawdown",  f"{metrics['max_drawdown']:.2%}")
col5.metric("⚠️ VaR (95%)",     f"{metrics['var_95']:.2%}")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🌪️ Scenario Analysis", "🔗 Correlations & Risk", "⚖️ Rebalancing"])

# ── TAB 1: PERFORMANCE ──
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("Cumulative Portfolio Return")
        cumulative = (1 + port_ret).cumprod()
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
        ax.set_facecolor('#0f1117')
        ax.plot(cumulative.index, cumulative.values, color='#60a5fa', linewidth=2)
        ax.fill_between(cumulative.index, 1, cumulative.values,
                        where=cumulative.values >= 1, alpha=0.15, color='#34d399')
        ax.fill_between(cumulative.index, 1, cumulative.values,
                        where=cumulative.values < 1,  alpha=0.15, color='#f87171')
        ax.axhline(1, color='#374151', linewidth=0.8, linestyle='--')
        ax.tick_params(colors='#9ca3af')
        ax.spines[:].set_color('#1e2130')
        ax.set_ylabel("Cumulative Return", color='#9ca3af')
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Top 10 Holdings")
        top10 = weights.nlargest(10).reset_index()
        top10.columns = ['Asset', 'Weight']
        top10['Sector'] = top10['Asset'].map(SECTORS)
        top10['Weight'] = top10['Weight'].apply(lambda x: f"{x:.2%}")
        st.dataframe(top10, use_container_width=True, hide_index=True)

    st.subheader("Sector Allocation")
    sector_weights = {}
    for asset, w in weights.items():
        s = SECTORS.get(asset, "Other")
        sector_weights[s] = sector_weights.get(s, 0) + w
    fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor='#0f1117')
    ax2.set_facecolor('#0f1117')
    sectors = list(sector_weights.keys())
    vals    = list(sector_weights.values())
    colors  = plt.cm.Set2(np.linspace(0, 1, len(sectors)))
    bars = ax2.barh(sectors, vals, color=colors, height=0.6)
    ax2.tick_params(colors='#9ca3af')
    ax2.spines[:].set_color('#1e2130')
    ax2.set_xlabel("Weight", color='#9ca3af')
    for bar, val in zip(bars, vals):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.1%}', va='center', color='#9ca3af', fontsize=9)
    st.pyplot(fig2)
    plt.close()

# ── TAB 2: SCENARIO ANALYSIS ──
with tab2:
    st.subheader("Market Shock Scenario Analysis")
    st.markdown("*Simulating market shocks from -10% to -30% and stress-testing portfolio risk exposure*")

    scenarios_df = run_scenario_analysis(weights, prices[assets])
    if save_to_db:
        save_scenarios(scenarios_df)

    display_df = scenarios_df[['Scenario', 'Market Shock', 'Portfolio P&L', 'Portfolio Value', 'Return']].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("P&L Under Each Scenario")
    fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
    ax3.set_facecolor('#0f1117')
    colors_bar = ['#34d399' if v >= 0 else '#f87171' for v in scenarios_df['raw_pnl']]
    bars = ax3.bar(scenarios_df['Scenario'], scenarios_df['raw_pnl'] / 1000, color=colors_bar, width=0.6)
    ax3.axhline(0, color='#374151', linewidth=0.8)
    ax3.tick_params(colors='#9ca3af', axis='both')
    ax3.spines[:].set_color('#1e2130')
    ax3.set_ylabel("P&L ($000s)", color='#9ca3af')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    st.pyplot(fig3)
    plt.close()

# ── TAB 3: CORRELATIONS & RISK ──
with tab3:
    col_x, col_y = st.columns(2)

    with col_x:
        st.subheader("Correlation Matrix (Top 20 Assets)")
        top20   = weights.nlargest(20).index.tolist()
        corr    = correlation_matrix(returns[top20])
        fig4, ax4 = plt.subplots(figsize=(8, 7), facecolor='#0f1117')
        ax4.set_facecolor('#0f1117')
        im = ax4.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(top20)))
        ax4.set_yticks(range(len(top20)))
        ax4.set_xticklabels(top20, rotation=90, fontsize=7, color='#9ca3af')
        ax4.set_yticklabels(top20, fontsize=7, color='#9ca3af')
        plt.colorbar(im, ax=ax4)
        ax4.spines[:].set_color('#1e2130')
        st.pyplot(fig4)
        plt.close()

    with col_y:
        st.subheader("Risk Contributions")
        rc = risk_contributions(weights, returns[assets])
        top_rc = rc.nlargest(15).reset_index()
        top_rc.columns = ['Asset', 'Risk Contribution']
        top_rc['Sector'] = top_rc['Asset'].map(SECTORS)
        fig5, ax5 = plt.subplots(figsize=(6, 6), facecolor='#0f1117')
        ax5.set_facecolor('#0f1117')
        colors5 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_rc)))
        ax5.barh(top_rc['Asset'], top_rc['Risk Contribution'], color=colors5, height=0.6)
        ax5.tick_params(colors='#9ca3af')
        ax5.spines[:].set_color('#1e2130')
        ax5.set_xlabel("Risk Contribution", color='#9ca3af')
        ax5.set_title("Top 15 Risk Contributors", color='#e5e7eb', fontsize=11)
        st.pyplot(fig5)
        plt.close()
        st.markdown("*High concentration in top assets signals diversification gaps*")

# ── TAB 4: REBALANCING ──
with tab4:
    st.subheader("Rebalancing Strategy Comparison")
    st.markdown("*Evaluating strategies that reduce portfolio volatility by up to 15%*")

    rebal_df, vol_reduction = compare_rebalancing(prices[assets], assets)
    st.dataframe(rebal_df[['Strategy','Ann. Return','Volatility','Sharpe','Max DD','Vol Reduction']],
                 use_container_width=True, hide_index=True)

    st.success(f"✅ Best rebalancing strategy reduced volatility by **{vol_reduction:.1%}** vs original portfolio")

    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0f1117')
    strategies = rebal_df['Strategy'].tolist()
    vols   = rebal_df['raw_vol'].tolist()
    colors6 = ['#f87171', '#60a5fa', '#34d399', '#fbbf24']

    for ax, vals, title, ylabel in [
        (axes6[0], vols, "Volatility by Strategy", "Annualized Volatility"),
    ]:
        ax.set_facecolor('#0f1117')
        ax.bar(strategies, vals, color=colors6, width=0.5)
        ax.tick_params(colors='#9ca3af')
        ax.spines[:].set_color('#1e2130')
        ax.set_ylabel(ylabel, color='#9ca3af')
        ax.set_title(title, color='#e5e7eb')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    sharpes = [float(s) for s in rebal_df['Sharpe'].tolist()]
    axes6[1].set_facecolor('#0f1117')
    axes6[1].bar(strategies, sharpes, color=colors6, width=0.5)
    axes6[1].tick_params(colors='#9ca3af')
    axes6[1].spines[:].set_color('#1e2130')
    axes6[1].set_ylabel("Sharpe Ratio", color='#9ca3af')
    axes6[1].set_title("Sharpe Ratio by Strategy", color='#e5e7eb')
    plt.setp(axes6[1].get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()
# Streamlit interactive dashboard
