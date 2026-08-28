"""
LoanScope — Intelligent Loan Performance Intelligence Engine
=============================================================
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
    page_title="LoanScope — Loan Performance Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, consistent typography and card styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .feature-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38bdf8;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-blue {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("## **LoanScope Engine**")
    st.markdown("*Quantitative Loan Surveillance & Risk Intelligence*")
    st.markdown("---")
    
    st.info(
        "🔬 **Hosted Demo Mode**\n\n"
        "Operating on a representative sample dataset for responsive cloud exploration. "
        "Full-scale run (**50,000 loans × 874K records**) can be executed locally via `make run-all`.",
        icon="ℹ️"
    )
    
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [GitHub Repository](https://github.com/ac265640/LoanScope)")
    st.markdown("- [Model Card (`reports/model_card.md`)](https://github.com/ac265640/LoanScope/blob/main/reports/model_card.md)")
    st.markdown("- [Validation Rules](https://github.com/ac265640/LoanScope/blob/main/data/validation_rules.json)")
    st.caption("Version 1.2.0 | Production Release")

# Main Page Body
st.markdown('<div class="main-header">🏦 LoanScope: Loan Performance Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An institutional-grade, multi-horizon credit risk, survival modeling, anomaly detection, and macroeconomic stress testing suite.</div>',
    unsafe_allow_html=True,
)

# Headline KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="🎯 3M Delinquency ROC-AUC", value="0.7977", delta="+0.0197 vs LR")
with col2:
    st.metric(label="📈 PR-AUC (3M Early Warning)", value="0.4090", delta="10.1x Base Prev.")
with col3:
    st.metric(label="⏳ Competing-Risk CIF Bias", value="+8.72 pp", delta="KM Overestimation", delta_color="inverse")
with col4:
    st.metric(label="🛡️ Anomaly Detection AUC", value="0.8310", delta="100% Rule Match")
with col5:
    st.metric(label="🧪 Automated Tests", value="100% Pass", delta="Zero Data Leakage")

st.markdown("---")

st.markdown("### 🧭 Interactive Showcase Navigation")
st.markdown("Select a module from the **left sidebar** or explore the feature sections below:")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    with st.container():
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📊 1. System Overview & Architecture</div>
                <p>Complete pipeline lineage, zero-leakage cohort partitioning, feature store specifications, and end-to-end quantitative scorecard.</p>
                <span class="badge-blue">Architecture</span> &nbsp; <span class="badge-green">Data Lineage</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🎯 2. Predictive Models & Probability Calibration</div>
                <p>Multi-outcome LightGBM classifiers (3M/6M delinquency, 12M default & prepayment), Platt sigmoid calibration, reliability diagrams, and decision threshold optimization ($t=0.50$ vs optimal $t^*$).</p>
                <span class="badge-blue">LightGBM</span> &nbsp; <span class="badge-green">Platt Scaling</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">⏳ 3. Survival Analysis & Competing Risks</div>
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
                <div class="feature-title">🛡️ 4. Anomaly Detection & Reviewer Cases</div>
                <p>Component A (deterministic Rule Engine VR001–VR005) + Component B (Isolation Forest + Learned ML). Explore 25 reviewer-ready anomaly cases with SHAP driver attributions and audit notes.</p>
                <span class="badge-blue">Isolation Forest</span> &nbsp; <span class="badge-green">25 Audit Cases</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">⚡ 5. Macroeconomic Scenario & Stress Simulator</div>
                <p>Multi-scenario stress simulations (Base, Adverse Credit +150bps, High Prepayment -75bps), segment vulnerability curves, and 1,000-path Monte Carlo stochastic risk distributions.</p>
                <span class="badge-blue">Stress Testing</span> &nbsp; <span class="badge-green">1,000 Monte Carlo Paths</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📐 6. Feature Drift Surveillance Dashboard</div>
                <p>Automated Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistic tracking between train and test distributions with pass/warn/fail thresholds.</p>
                <span class="badge-blue">PSI & KS</span> &nbsp; <span class="badge-green">Live Drift Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("#### 🔒 Governance & Ethical AI Statement")
st.caption(
    "All models are calibrated and audited for subgroup fairness (Four-Fifths Rule compliance). "
    "Counterfactual levers provide adverse action remediation. All outputs and copilot notes are strictly advisory recommendations — not automated credit denials."
)
