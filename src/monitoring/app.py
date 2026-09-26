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

import sys
from pathlib import Path
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _theme import inject_theme  # noqa: E402

# Page configuration — wrapped so this file still works standalone
# (`streamlit run src/monitoring/app.py`) AND when hosted as a page under
# st.navigation() from drift_dashboard.py, where set_page_config is already
# called once by the parent entry point.
try:
    st.set_page_config(
        page_title="LoanScope — Quantitative Risk & Surveillance Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

inject_theme()

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

# Headline KPI Cards — bordered card grid, vertically centered, equal height
kpi_metrics = [
    dict(label="3M Delinquency ROC-AUC", value="0.7977", delta="+0.0197 vs LR"),
    dict(label="3M Early Warning PR-AUC", value="0.4090", delta="10.1x Base Prevalence"),
    dict(label="Competing-Risk CIF Bias", value="+8.72 pp", delta="KM Overestimation Removed", delta_color="inverse"),
    dict(label="Anomaly Detection ROC-AUC", value="0.8310", delta="100% Rule Engine Match"),
    dict(label="Automated Pipeline Tests", value="100% Pass", delta="Zero Data Leakage"),
]
kpi_cols = st.columns(5, vertical_alignment="center")
for col, kpi in zip(kpi_cols, kpi_metrics):
    with col:
        with st.container(border=True, height="stretch"):
            st.metric(**kpi)

st.markdown("---")

st.markdown("### Platform Modules")
st.markdown("Select a module from the left navigation menu or explore the platform sections below:")

# Data-driven feature grid — dynamic 2-column layout built via enumerate()
PLATFORM_MODULES = [
    {
        "title": "1. System Overview & Architecture",
        "body": "Pipeline data lineage, zero-leakage cohort partitioning (778K train / 95K val / 69K test), feature store specifications, and performance scorecard.",
        "badges": [("badge-blue", "Architecture"), ("badge-green", "Data Lineage")],
    },
    {
        "title": "2. Predictive Models & Probability Calibration",
        "body": "Multi-outcome LightGBM classifiers (3M/6M delinquency, 12M default, 12M prepayment), Platt sigmoid scaling, reliability diagrams, and decision threshold optimization.",
        "badges": [("badge-blue", "LightGBM"), ("badge-green", "Platt Scaling")],
    },
    {
        "title": "3. Survival Analysis & Competing Risks",
        "body": "Cause-specific Aalen-Johansen Cumulative Incidence Functions (CIF) modeling default and voluntary prepayment as competing terminal events, eliminating naive Kaplan-Meier overestimation bias (+8.72pp).",
        "badges": [("badge-blue", "Aalen-Johansen"), ("badge-green", "Competing Risks")],
    },
    {
        "title": "4. Anomaly Detection & Reviewer Cases",
        "body": "Component A (Deterministic Rule Engine VR001–VR005) + Component B (Isolation Forest + Learned ML). 25 reviewer-ready anomaly cases with SHAP driver attributions and structured audit notes.",
        "badges": [("badge-blue", "Isolation Forest"), ("badge-green", "25 Reviewer Cases")],
    },
    {
        "title": "5. Macroeconomic Scenario & Stress Simulator",
        "body": "Multi-scenario stress projections (Base, Adverse Credit +150bps, High Prepayment -75bps), segment vulnerability curves, and 1,000-path Monte Carlo stochastic risk distributions.",
        "badges": [("badge-blue", "Stress Testing"), ("badge-green", "1,000 Monte Carlo Paths")],
    },
    {
        "title": "6. Feature Drift Surveillance Dashboard",
        "body": "Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistic tracking between historical train and out-of-time test distributions with formal thresholds.",
        "badges": [("badge-blue", "PSI & KS"), ("badge-green", "Drift Surveillance")],
    },
]

N_GRID_COLS = 2
grid_cols = st.columns(N_GRID_COLS)
for i, module in enumerate(PLATFORM_MODULES):
    col = grid_cols[i % N_GRID_COLS]
    with col:
        badge_html = " &nbsp; ".join(f'<span class="{cls}">{label}</span>' for cls, label in module["badges"])
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-title">{module["title"]}</div>
                <p>{module["body"]}</p>
                {badge_html}
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
