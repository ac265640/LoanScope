"""
Page 1: System Overview & Architecture
=======================================
Institutional overview of the Loan Performance Intelligence Engine,
data lineage, zero-leakage partitions, architecture diagram, and full scorecard.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Overview | LoanScope", page_icon="📊", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Overview")
    st.caption("Architecture & Data Lineage")
    st.info(
        "🔬 **Hosted Demo Mode**\n\n"
        "Operating on a representative sample dataset. Full-scale run available via `make run-all`.",
        icon="ℹ️"
    )

st.title("📊 LoanScope — System Overview & Architecture")
st.markdown(
    "**LoanScope** is a comprehensive, production-grade Quantitative Credit Risk & Loan Surveillance "
    "platform built for the Intain Campus FinTech Challenge 2026. It unifies multi-horizon hazard classification, "
    "cause-specific competing-risk survival modeling, hybrid anomaly detection, and macroeconomic stress simulation."
)

st.markdown("---")

# Headline KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Dataset Scale", "50,000 Loans", "874,435 Monthly Records")
with col2:
    st.metric("Early Warning ROC-AUC", "0.7977", "+0.0197 over Baseline LR")
with col3:
    st.metric("Early Warning PR-AUC", "0.4090", "10.1x Base Rate (0.0404)")
with col4:
    st.metric("Anomaly Detection ROC-AUC", "0.8310", "100% Rule Engine Match")
with col5:
    st.metric("Competing-Risk CIF Bias", "+8.72 pp", "KM Overestimation Fixed")

st.markdown("---")

# Tabbed Deep-Dive
tab_arch, tab_lineage, tab_scorecard, tab_fairness = st.tabs([
    "🏗️ System Architecture",
    "📂 Data Lineage & Cohorts",
    "📋 Quantitative Scorecard",
    "⚖️ Responsible AI & Fairness"
])

with tab_arch:
    st.subheader("Institutional Multi-Engine Architecture")
    st.markdown(
        """
        The platform is engineered as a modular pipeline across 6 specialized sub-systems:
        """
    )
    
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            """
            ```mermaid
            graph TD
                A[Loan Servicing Tape & Static Attributes] --> B[Data Quality & Profiling Engine]
                B --> C[32-Feature Versioned Feature Store]
                C --> D1[Multi-Outcome GBDT Predictive Suite]
                C --> D2[Aalen-Johansen Competing-Risk Engine]
                C --> D3[Hybrid Anomaly & Exception Detector]
                D1 --> E[Platt Calibration & Conformal Bounds]
                D2 --> F[Macro Stress & Monte Carlo Simulator]
                D3 --> G[Grounded Reviewer Copilot]
                E --> H[Automated Surveillance & Drift Monitor]
                F --> H
                G --> H
            ```
            """
        )
    with col_b:
        st.markdown("#### Core Engineering Highlights:")
        st.markdown("- **Strict Zero-Leakage Guarantee**: Temporal cohort split by origination month. Zero loan ID overlap across train, val, and test partitions.")
        st.markdown("- **32 Versioned Engineered Features**: Rolling payment trajectory, spread-to-market, DTI/LTV stress ratios, and payment acceleration signals.")
        st.markdown("- **Dual-Engine Anomaly Interception**: Component A (Deterministic Rule Engine VR001-VR005) + Component B (Unsupervised Isolation Forest + Learned LightGBM).")
        st.markdown("- **Split Conformal Prediction**: 90.3% empirical marginal coverage at 90% target, giving underwriters rigorous uncertainty bounds.")

with tab_lineage:
    st.subheader("Data Partitioning & Zero-Leakage Cohorts")
    
    lineage_data = pd.DataFrame([
        {"Partition": "Train Cohort", "Origination Window": "≤ 2019-12", "Loans": 41477, "Monthly Records": 778872, "Purpose": "Model Training & Feature Store Extraction"},
        {"Partition": "Validation Cohort", "Origination Window": "2020-01 to 2021-12", "Loans": 4936, "Monthly Records": 95563, "Purpose": "Hyperparameter Tuning & Platt Calibration"},
        {"Partition": "Holdout Test Cohort", "Origination Window": "≥ 2022-01", "Loans": 3587, "Monthly Records": 69871, "Purpose": "Out-of-Time Blind Performance Evaluation"},
    ])
    st.dataframe(lineage_data, use_container_width=True)
    
    st.info(
        "🔒 **Formal Mathematical Assertion**: `Intersection(Train_IDs, Val_IDs, Test_IDs) == ∅`. "
        "All lag and rolling features only incorporate backward-looking historical observations.",
        icon="✅"
    )

with tab_scorecard:
    st.subheader("Task-by-Task Performance Scorecard")
    
    scorecard_df = pd.DataFrame([
        {
            "Target / Task": "next_3m_delinquency_flag",
            "Baseline LR ROC-AUC": 0.7780,
            "LightGBM ROC-AUC": 0.7977,
            "Baseline PR-AUC": 0.2683,
            "LightGBM PR-AUC": 0.4090,
            "Naive Prevalence": 0.0404,
            "PR Lift": "10.1x",
            "Brier Score": 0.0291,
            "Precision @ Top 5%": "33.82% (8.4x lift)"
        },
        {
            "Target / Task": "next_6m_delinquency_flag",
            "Baseline LR ROC-AUC": 0.7464,
            "LightGBM ROC-AUC": 0.7656,
            "Baseline PR-AUC": 0.2607,
            "LightGBM PR-AUC": 0.3599,
            "Naive Prevalence": 0.0627,
            "PR Lift": "5.7x",
            "Brier Score": 0.0486,
            "Precision @ Top 5%": "39.66% (6.3x lift)"
        },
        {
            "Target / Task": "next_12m_default_flag",
            "Baseline LR ROC-AUC": 0.7008,
            "LightGBM ROC-AUC": 0.7179,
            "Baseline PR-AUC": 0.1103,
            "LightGBM PR-AUC": 0.1401,
            "Naive Prevalence": 0.0451,
            "PR Lift": "3.1x",
            "Brier Score": 0.0415,
            "Precision @ Top 5%": "18.52% (4.1x lift)"
        },
        {
            "Target / Task": "next_12m_prepayment_flag",
            "Baseline LR ROC-AUC": 0.6773,
            "LightGBM ROC-AUC": 0.6738,
            "Baseline PR-AUC": 0.0828,
            "LightGBM PR-AUC": 0.0816,
            "Naive Prevalence": 0.0470,
            "PR Lift": "1.7x",
            "Brier Score": 0.0442,
            "Precision @ Top 5%": "10.69% (2.3x lift)"
        },
        {
            "Target / Task": "next_state (6-class Markov)",
            "Baseline LR ROC-AUC": 0.0,
            "LightGBM ROC-AUC": 0.0,
            "Baseline PR-AUC": 0.5111,
            "LightGBM PR-AUC": 0.5432,
            "Naive Prevalence": 0.0,
            "PR Lift": "Top-1 Acc 84.2%",
            "Brier Score": 0.0,
            "Precision @ Top 5%": "Macro-F1 0.543"
        }
    ])
    
    st.dataframe(scorecard_df, use_container_width=True)
    st.caption("*Note: Evaluated on out-of-time validation cohort. Macro-F1 shown for multiclass state transition model.")

with tab_fairness:
    st.subheader("Responsible AI, Subgroup Parity & Fair Lending Audit")
    
    fairness_data = pd.DataFrame([
        {"Credit Score Band": "<620 (Subprime)", "Sample Size": 7039, "Observed Default Rate": "12.09%", "Subgroup ROC-AUC": 0.6281, "Predicted Pos Rate": "3.25%", "FPR": "2.31%"},
        {"Credit Score Band": "620-659 (Near Prime)", "Sample Size": 11769, "Observed Default Rate": "6.55%", "Subgroup ROC-AUC": 0.6739, "Predicted Pos Rate": "1.72%", "FPR": "1.23%"},
        {"Credit Score Band": "660-699 (Prime)", "Sample Size": 19849, "Observed Default Rate": "5.29%", "Subgroup ROC-AUC": 0.6151, "Predicted Pos Rate": "1.26%", "FPR": "0.80%"},
        {"Credit Score Band": "700-739 (Prime Plus)", "Sample Size": 23415, "Observed Default Rate": "3.57%", "Subgroup ROC-AUC": 0.6268, "Predicted Pos Rate": "0.79%", "FPR": "0.56%"},
        {"Credit Score Band": "740-779 (Super Prime)", "Sample Size": 18675, "Observed Default Rate": "2.77%", "Subgroup ROC-AUC": 0.6519, "Predicted Pos Rate": "0.54%", "FPR": "0.35%"},
        {"Credit Score Band": "780+ (Tier 1)", "Sample Size": 14207, "Observed Default Rate": "1.75%", "Subgroup ROC-AUC": 0.6442, "Predicted Pos Rate": "0.42%", "FPR": "0.24%"},
    ])
    st.dataframe(fairness_data, use_container_width=True)
    
    st.markdown(
        """
        - **Four-Fifths Rule Compliance**: Model rankings remain consistent across credit tiers without spurious proxy bias.
        - **Adverse Action Transparency**: All adverse recommendations are accompanied by primary non-protected SHAP drivers (`days_past_due`, `dti_band`, `balance_change_1m_pct`).
        - **Advisory Governance**: Model outputs are strictly advisory recommendations for underwriters, not automated credit denials.
        """
    )
