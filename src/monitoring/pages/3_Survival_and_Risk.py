"""
Page 3: Survival Analysis & Competing Risks
============================================
Cause-specific Aalen-Johansen Cumulative Incidence Functions (CIF) vs
naive Kaplan-Meier single-risk overestimation (+8.72pp bias).
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

_THIS_DIR = Path(__file__).resolve().parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _theme import inject_theme, apply_chart_theme  # noqa: E402

try:
    st.set_page_config(page_title="Survival & Risk | LoanScope", layout="wide")
except Exception:
    pass

inject_theme()

# Sidebar
with st.sidebar:
    st.markdown("### Survival Analysis")
    st.caption("Aalen-Johansen Competing Risks")
    st.info(
        "**Hosted Demo Mode**\n\n"
        "Pre-computed survival analytics from portfolio event history."
    )

st.title("Survival Analysis & Competing-Risk CIF Modeling")
st.markdown(
    "In mortgage and consumer debt portfolios, **voluntary prepayment** and **involuntary default** "
    "are mutually competing terminal events. When a borrower prepays in full, they can no longer default."
)

st.markdown("---")

# Key Metrics — bordered, vertically centered, equal-height card grid
c1, c2, c3, c4 = st.columns(4, vertical_alignment="center")
with c1:
    with st.container(border=True, height="stretch"):
        st.metric("36M Single-Risk Default", "23.13%", "Naive Kaplan-Meier")
with c2:
    with st.container(border=True, height="stretch"):
        st.metric("36M Competing-Risk Default", "14.41%", "Aalen-Johansen CIF")
with c3:
    with st.container(border=True, height="stretch"):
        st.metric("KM Overestimation Bias", "+8.72 pp", "Eliminated Capital Penalty", delta_color="inverse")
with c4:
    with st.container(border=True, height="stretch"):
        st.metric("36M Prepayment CIF", "8.47%", "Voluntary Refinance")

st.markdown("---")

col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.subheader("Cumulative Incidence Functions (Aalen-Johansen CIF)")
    
    # Horizons and CIF data from reports/survival_report.md
    months = [0, 6, 12, 18, 24, 30, 36]
    cif_default = [0.0, 0.038, 0.0847, 0.110, 0.1315, 0.138, 0.1441]
    cif_prepay = [0.0, 0.029, 0.0631, 0.072, 0.0793, 0.082, 0.0847]
    km_naive_default = [0.0, 0.041, 0.0889, 0.132, 0.1738, 0.201, 0.2313]
    
    fig = go.Figure()
    
    # Naive KM Default
    fig.add_trace(go.Scatter(
        x=months, y=km_naive_default,
        mode='lines+markers',
        name='Naive Single-Risk KM (Default)',
        line=dict(color='#ef4444', dash='dash', width=2),
        marker=dict(size=6)
    ))
    
    # True Competing-Risk CIF Default
    fig.add_trace(go.Scatter(
        x=months, y=cif_default,
        mode='lines+markers',
        name='Aalen-Johansen CIF (Default)',
        line=dict(color='#f97316', width=3),
        marker=dict(size=8)
    ))
    
    # True Competing-Risk CIF Prepay
    fig.add_trace(go.Scatter(
        x=months, y=cif_prepay,
        mode='lines+markers',
        name='Aalen-Johansen CIF (Prepaid)',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=8)
    ))
    
    fig = apply_chart_theme(fig, height=470, title="Cumulative Incidence Over Time: Competing-Risk CIF vs Naive KM")
    fig.update_layout(
        xaxis_title="Months Since Origination",
        yaxis_title="Cumulative Event Probability",
        yaxis=dict(tickformat=".1%"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Competing Risk Dynamics in Portfolio Surveillance")
    st.markdown(
        """
        Standard single-risk survival models (like naive Kaplan-Meier) treat voluntary payoffs 
        as **random censoring** (as if the prepaid borrower were still at risk of defaulting in the future).
        
        **The Problem**:
        - Artificially inflates the default hazard at longer durations.
        - Produces an **+8.72 percentage point overestimation** of 36-month default risk (23.13% vs 14.41%).
        
        **Institutional Impact**:
        - Eliminates phantom credit reserve requirements.
        - Provides accurate duration estimates for loan portfolio valuation and securitization tranche pricing.
        """
    )
    
    @st.cache_data(ttl="6h", show_spinner=False)
    def _load_bias_df() -> pd.DataFrame:
        return pd.DataFrame([
            {"Horizon": "12 Months", "CIF Default": "8.47%", "Naive KM Default": "8.89%", "Overestimation Bias": "+0.43 pp"},
            {"Horizon": "24 Months", "CIF Default": "13.15%", "Naive KM Default": "17.38%", "Overestimation Bias": "+4.23 pp"},
            {"Horizon": "36 Months", "CIF Default": "14.41%", "Naive KM Default": "23.13%", "Overestimation Bias": "+8.72 pp"},
        ])

    bias_df = _load_bias_df()
    st.dataframe(bias_df, use_container_width=True)

st.markdown("---")

st.subheader("Cumulative Incidence Rates by Credit Score Band")


@st.cache_data(ttl="6h", show_spinner=False)
def _load_credit_cif_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Credit Band": "<620 (Subprime)", "12M CIF Default": "19.57%", "12M CIF Prepayment": "0.00%", "36M CIF Default": "32.61%", "36M CIF Prepayment": "0.00%", "Primary Risk": "Severe Credit Default"},
        {"Credit Band": "620-659 (Near Prime)", "12M CIF Default": "16.90%", "12M CIF Prepayment": "1.41%", "36M CIF Default": "28.17%", "36M CIF Prepayment": "1.41%", "Primary Risk": "Elevated Default"},
        {"Credit Band": "660-699 (Prime)", "12M CIF Default": "10.00%", "12M CIF Prepayment": "1.11%", "36M CIF Default": "18.89%", "36M CIF Prepayment": "2.22%", "Primary Risk": "Moderate Default"},
        {"Credit Band": "700-739 (Prime Plus)", "12M CIF Default": "3.94%", "12M CIF Prepayment": "7.87%", "36M CIF Default": "4.72%", "36M CIF Prepayment": "8.66%", "Primary Risk": "Balanced / Low Risk"},
        {"Credit Band": "740-779 (Super Prime)", "12M CIF Default": "3.06%", "12M CIF Prepayment": "7.14%", "36M CIF Default": "10.20%", "36M CIF Prepayment": "11.22%", "Primary Risk": "Refinance Flight Risk"},
        {"Credit Band": "780+ (Top Tier)", "12M CIF Default": "2.50%", "12M CIF Prepayment": "17.50%", "36M CIF Default": "5.00%", "36M CIF Prepayment": "22.50%", "Primary Risk": "High Prepayment (Duration Risk)"},
    ])


credit_cif_df = _load_credit_cif_df()

st.dataframe(credit_cif_df, use_container_width=True)
st.caption("Data source: reports/survival_report.md | Estimator: Aalen-Johansen non-parametric competing risk model.")
