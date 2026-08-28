"""
LoanScope — Loan Performance Intelligence Engine
=================================================
Main Showcase Application Entrypoint for Streamlit Community Cloud.

Multi-Page Structure:
- 1_Overview.py: System architecture, pipeline lineage, headline KPIs
- 2_Predictions.py: Multi-horizon GBDT models, calibration diagrams, threshold trade-offs
- 3_Survival_and_Risk.py: Competing risks Aalen-Johansen CIF vs Kaplan-Meier
- 4_Anomaly_Cases.py: 25 reviewer cases, Rule Engine vs Learned ML, audit notes
- 5_Scenario_Simulator.py: Macro stress scenarios & Monte Carlo fan charts
- 6_Drift_Monitoring.py: Feature distribution stability (PSI & KS metrics)
"""

from pathlib import Path
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]

# Page configuration
st.set_page_config(
    page_title="LoanScope — Quantitative Risk & Surveillance Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, institutional typography and card styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
    }
    .feature-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #38bdf8;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global Sidebar
with st.sidebar:
    st.markdown("## **LoanScope Platform**")
    st.caption("Quantitative Loan Surveillance & Risk Analytics")
    st.markdown("---")
    
    st.info(
        "**Hosted Demo Mode**\n\n"
        "Operating on a representative sample dataset for cloud evaluation. "
        "Full-scale execution (50,000 loans × 874,435 records) available via `make run-all`."
    )
    
    st.markdown("### System Documentation")
    st.markdown("- [GitHub Repository](https://github.com/ac265640/LoanScope)")
    st.markdown("- [Model Card (reports/model_card.md)](https://github.com/ac265640/LoanScope/blob/main/reports/model_card.md)")
    st.markdown("- [Validation Rules](https://github.com/ac265640/LoanScope/blob/main/data/validation_rules.json)")
    st.caption("Version 1.2.0 | Production Release")

# Main Page Body
st.markdown('<div class="main-header">LoanScope: Loan Performance Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional multi-horizon credit risk prediction, cause-specific survival modeling, anomaly detection, and macroeconomic stress testing suite.</div>',
    unsafe_allow_html=True,
)

# Headline KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="3M Delinquency ROC-AUC", value="0.7977", delta="+0.0197 vs LR")
with col2:
    st.metric(label="3M Early Warning PR-AUC", value="0.4090", delta="10.1x Base Prevalence")
with col3:
    st.metric(label="Competing-Risk CIF Bias", value="+8.72 pp", delta="KM Overestimation Removed", delta_color="inverse")
with col4:
    st.metric(label="Anomaly Detection ROC-AUC", value="0.8310", delta="100% Rule Engine Match")
with col5:
    st.metric(label="Automated Pipeline Tests", value="100% Pass", delta="Zero Data Leakage")

st.markdown("---")

st.markdown("### Platform Modules")
st.markdown("Select a module from the left navigation menu or explore the platform sections below:")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    with st.container():
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">1. System Overview & Architecture</div>
                <p>Pipeline data lineage, zero-leakage cohort partitioning (778K train / 95K val / 69K test), feature store specifications, and performance scorecard.</p>
                <span class="badge-blue">Architecture</span> &nbsp; <span class="badge-green">Data Lineage</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">2. Predictive Models & Probability Calibration</div>
                <p>Multi-outcome LightGBM classifiers (3M/6M delinquency, 12M default, 12M prepayment), Platt sigmoid scaling, reliability diagrams, and decision threshold optimization.</p>
                <span class="badge-blue">LightGBM</span> &nbsp; <span class="badge-green">Platt Scaling</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">3. Survival Analysis & Competing Risks</div>
                <p>Cause-specific Aalen-Johansen Cumulative Incidence Functions (CIF) modeling default and voluntary prepayment as competing terminal events, eliminating naive Kaplan-Meier overestimation bias (+8.72pp).</p>
                <span class="badge-blue">Aalen-Johansen</span> &nbsp; <span class="badge-green">Competing Risks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with nav_col2:
    with st.container():
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">4. Anomaly Detection & Reviewer Cases</div>
                <p>Component A (Deterministic Rule Engine VR001–VR005) + Component B (Isolation Forest + Learned ML). 25 reviewer-ready anomaly cases with SHAP driver attributions and structured audit notes.</p>
                <span class="badge-blue">Isolation Forest</span> &nbsp; <span class="badge-green">25 Reviewer Cases</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">5. Macroeconomic Scenario & Stress Simulator</div>
                <p>Multi-scenario stress projections (Base, Adverse Credit +150bps, High Prepayment -75bps), segment vulnerability curves, and 1,000-path Monte Carlo stochastic risk distributions.</p>
                <span class="badge-blue">Stress Testing</span> &nbsp; <span class="badge-green">1,000 Monte Carlo Paths</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">6. Feature Drift Surveillance Dashboard</div>
                <p>Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistic tracking between historical train and out-of-time test distributions with formal thresholds.</p>
                <span class="badge-blue">PSI & KS</span> &nbsp; <span class="badge-green">Drift Surveillance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("#### Governance & Responsible AI Standards")
st.caption(
    "Models are calibrated and audited for subgroup fairness (Four-Fifths Rule compliance). "
    "Counterfactual levers provide adverse action remediation guidance. All outputs and copilot notes are strictly advisory recommendations for underwriting analysts."
)
