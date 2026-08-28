"""
Page 5: Scenario Simulator & Monte Carlo Risk
=============================================
Macroeconomic scenario stress testing (Base, Adverse Credit, High Prepayment),
segment-level vulnerability curves, and 1,000-path Monte Carlo risk distributions.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Scenario Simulator | LoanScope", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### Scenario Simulator")
    st.caption("Stress Testing & Monte Carlo")
    st.info(
        "**Hosted Demo Mode**\n\n"
        "Pre-computed 1,000-path Monte Carlo simulations from reports/scenario_report.md."
    )

st.title("Macroeconomic Scenario Stress Simulator & Monte Carlo")
st.markdown(
    "Evaluation of loan portfolio resilience under macroeconomic rate, unemployment, and housing price shocks. "
    "Features deterministic scenario projections alongside **1,000-path Monte Carlo stochastic risk distributions**."
)

st.markdown("---")

# Scenario Definitions Table
st.subheader("1. Macroeconomic Scenario Definitions & Shocks")
scenarios_df = pd.DataFrame([
    {
        "Scenario": "base",
        "Rate Shock": "0 bps",
        "Unemployment Δ": "+0.0%",
        "Home Price (HPI) Δ": "+0.0%",
        "Default Multiplier": "1.0x",
        "Prepay Multiplier": "1.0x",
        "Macro Narrative": "Steady macro environment with baseline employment and rate stability."
    },
    {
        "Scenario": "adverse_credit",
        "Rate Shock": "+150 bps",
        "Unemployment Δ": "+2.5 pp",
        "Home Price (HPI) Δ": "-8.0%",
        "Default Multiplier": "2.2x",
        "Prepay Multiplier": "0.6x",
        "Macro Narrative": "Severe stagflationary credit shock: rate spike, job losses, and collateral devaluation."
    },
    {
        "Scenario": "high_prepayment",
        "Rate Shock": "-75 bps",
        "Unemployment Δ": "-0.5 pp",
        "Home Price (HPI) Δ": "+5.0%",
        "Default Multiplier": "0.75x",
        "Prepay Multiplier": "2.5x",
        "Macro Narrative": "Refinance boom: rate drops trigger surge in voluntary mortgage prepayments."
    }
])
st.dataframe(scenarios_df, use_container_width=True)

st.markdown("---")

# Portfolio Projected Rates
st.subheader("2. Portfolio-Level Stress Loss & Prepayment Projections")

p_col1, p_col2 = st.columns([1.2, 1])

portfolio_rates = pd.DataFrame([
    {"Scenario": "Base", "3M Delinquency": 5.24, "6M Delinquency": 7.85, "12M Default": 5.39, "12M Prepayment": 4.51},
    {"Scenario": "Adverse Credit (+150bps)", "3M Delinquency": 13.13, "6M Delinquency": 20.36, "12M Default": 15.87, "12M Prepayment": 2.68},
    {"Scenario": "High Prepayment (-75bps)", "3M Delinquency": 4.19, "6M Delinquency": 6.28, "12M Default": 4.05, "12M Prepayment": 11.40},
])

with p_col1:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="12M Default Rate (%)", x=portfolio_rates["Scenario"], y=portfolio_rates["12M Default"], marker_color="#ef4444"))
    fig_bar.add_trace(go.Bar(name="12M Prepayment Rate (%)", x=portfolio_rates["Scenario"], y=portfolio_rates["12M Prepayment"], marker_color="#38bdf8"))
    fig_bar.add_trace(go.Bar(name="6M Delinquency Rate (%)", x=portfolio_rates["Scenario"], y=portfolio_rates["6M Delinquency"], marker_color="#f59e0b"))
    
    fig_bar.update_layout(
        title="Projected Rates by Macro Scenario",
        yaxis_title="Projected Portfolio Rate (%)",
        barmode='group',
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.05, y=0.95),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with p_col2:
    st.markdown("#### Scenario Comparison Table:")
    st.dataframe(portfolio_rates.style.format({
        "3M Delinquency": "{:.2f}%",
        "6M Delinquency": "{:.2f}%",
        "12M Default": "{:.2f}%",
        "12M Prepayment": "{:.2f}%"
    }), use_container_width=True)
    st.info(
        "**Key Analytical Finding**: Under Adverse Credit, default rates surge nearly **3x** (5.39% → 15.87%), "
        "while High Prepayment drives voluntary payoff rates to **11.40%** (shortening duration)."
    )

st.markdown("---")

# Segment Stress Vulnerability
st.subheader("3. Segment-Level Stress Vulnerability")
segment_view = st.radio("Select Segment Cut:", options=["Credit Score Band", "Origination Vintage Era"], horizontal=True)

if segment_view == "Credit Score Band":
    seg_df = pd.DataFrame([
        {"Credit Band": "<620 (Subprime)", "Base Default": 5.73, "Adverse Default": 16.24, "High Prepay Default": 4.30, "Stress Delta (Adverse vs Base)": "+10.51 pp"},
        {"Credit Band": "620-659 (Near Prime)", "Base Default": 5.49, "Adverse Default": 15.95, "High Prepay Default": 4.12, "Stress Delta (Adverse vs Base)": "+10.46 pp"},
        {"Credit Band": "660-699 (Prime)", "Base Default": 5.33, "Adverse Default": 15.65, "High Prepay Default": 4.00, "Stress Delta (Adverse vs Base)": "+10.32 pp"},
        {"Credit Band": "700-739 (Prime Plus)", "Base Default": 5.28, "Adverse Default": 15.60, "High Prepay Default": 3.96, "Stress Delta (Adverse vs Base)": "+10.32 pp"},
        {"Credit Band": "740-779 (Super Prime)", "Base Default": 5.22, "Adverse Default": 15.48, "High Prepay Default": 3.92, "Stress Delta (Adverse vs Base)": "+10.26 pp"},
        {"Credit Band": "780+ (Top Tier)", "Base Default": 5.10, "Adverse Default": 15.24, "High Prepay Default": 3.82, "Stress Delta (Adverse vs Base)": "+10.14 pp"},
    ])
    
    fig_seg = px.bar(
        seg_df, x="Credit Band", y=["Base Default", "Adverse Default", "High Prepay Default"],
        barmode='group',
        title="12M Default Rate by Credit Tier Across Scenarios (%)",
        color_discrete_map={"Base Default": "#94a3b8", "Adverse Default": "#ef4444", "High Prepay Default": "#38bdf8"},
        height=380
    )
    st.plotly_chart(fig_seg, use_container_width=True)
    st.dataframe(seg_df, use_container_width=True)

else:
    vintage_df = pd.DataFrame([
        {"Vintage Era": "Pre-2010 (Legacy)", "Base Default": 6.60, "Adverse Default": 19.66, "Base Prepayment": 4.36, "High Prepay Prepayment": 11.02},
        {"Vintage Era": "2011-2018 (Post-Crisis)", "Base Default": 4.55, "Adverse Default": 13.23, "Base Prepayment": 4.60, "High Prepay Prepayment": 11.64},
        {"Vintage Era": "2019+ (Recent)", "Base Default": 4.55, "Adverse Default": 13.19, "Base Prepayment": 4.65, "High Prepay Prepayment": 11.74},
    ])
    fig_vint = px.bar(
        vintage_df, x="Vintage Era", y=["Base Default", "Adverse Default"],
        barmode='group',
        title="12M Default Rate by Vintage Era (%)",
        color_discrete_map={"Base Default": "#94a3b8", "Adverse Default": "#ef4444"},
        height=380
    )
    st.plotly_chart(fig_vint, use_container_width=True)
    st.dataframe(vintage_df, use_container_width=True)

st.markdown("---")

# Monte Carlo Stochastic Distributions
st.subheader("4. Monte Carlo Portfolio Simulation (N = 1,000 Paths)")
st.markdown(
    "Stochastic simulation sampling calibrated Beta outcome distributions per loan across 1,000 macroeconomic paths, "
    "providing rigorous tail-risk quantiles (P1 to P99) for capital reserving."
)

mc_df = pd.DataFrame([
    {"Scenario": "Base", "P1": "7.97%", "P5": "8.15%", "P25": "8.09%", "Median (P50)": "8.15%", "P75": "8.21%", "P95": "8.30%", "P99 (Tail VaR)": "8.37%", "Std Dev": "0.09%"},
    {"Scenario": "Adverse Credit", "P1": "16.47%", "P5": "16.75%", "P25": "16.67%", "Median (P50)": "16.75%", "P75": "16.83%", "P95": "16.95%", "P99 (Tail VaR)": "17.02%", "Std Dev": "0.12%"},
    {"Scenario": "High Prepayment", "P1": "5.94%", "P5": "6.11%", "P25": "6.06%", "Median (P50)": "6.11%", "P75": "6.17%", "P95": "6.24%", "P99 (Tail VaR)": "6.29%", "Std Dev": "0.08%"},
])
st.dataframe(mc_df, use_container_width=True)
st.caption("Data source: reports/scenario_report.md | Module: src/scenarios/monte_carlo.py")
