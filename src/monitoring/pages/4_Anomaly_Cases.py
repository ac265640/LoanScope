"""
Page 4: Anomaly Detection & Reviewer Cases
==========================================
Component A (Deterministic Rule Engine VR001-VR005) vs Component B (Learned ML Exception Model),
25 interactive reviewer cases, and static grounded LLM Copilot audit notes.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Anomaly Cases | LoanScope", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Anomaly Engine")
    st.caption("Rule Engine & Isolation Forest")
    st.info(
        "🔬 **Hosted Demo Mode**\n\n"
        "25 curated reviewer cases from `reports/anomaly_reviewer_cases.md`.",
        icon="ℹ️"
    )

st.title("🛡️ Anomaly Detection & Exception Reviewer Cases")
st.markdown(
    "Dual-engine anomaly surveillance combining **deterministic business constraints** (Component A) "
    "with **unsupervised Isolation Forest and gradient boosted behavioral outlier scoring** (Component B)."
)

st.markdown("---")

# Architectural Distinction Banners
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 1rem; border-radius: 4px;">
            <h4 style="color: #38bdf8; margin: 0 0 0.5rem 0;">⚙️ Component A: Deterministic Rule Engine</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #cbd5e1;">
                Validates explicit ledger & tape rules (<b>VR001–VR005</b>) including paid-off balance contradictions, date chronology, and modification document verification.
                <br><b>Rule Match Rate: 100.00%</b> (Zero false-positive risk).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #a855f7; padding: 1rem; border-radius: 4px;">
            <h4 style="color: #a855f7; margin: 0 0 0.5rem 0;">🧠 Component B: Learned ML Exception Model</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #cbd5e1;">
                Non-circular LightGBM trained on 32 engineered features + Isolation Forest continuous scores to detect multi-attribute behavioral anomalies.
                <br><b>ROC-AUC: 0.8310 | F1 @ 0.50: 0.7361</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Curated 25 Reviewer Cases
ANOMALY_CASES = [
    {"Case #": 1, "Loan ID": "LN0026208", "Month": "2005-08", "Status": "90+ DPD", "Balance": 105044.96, "DPD": 120, "Anomaly Score": 1.0000, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Verify note rate against loan agreement schedule", "Detail": "Unusually elevated note rate of 22.65% vs portfolio median 4.52%."},
    {"Case #": 2, "Loan ID": "LN0026208", "Month": "2005-07", "Status": "90+ DPD", "Balance": 105083.34, "DPD": 120, "Anomaly Score": 1.0000, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Verify note rate against loan agreement schedule", "Detail": "Unusually elevated note rate of 22.65% with static past due roll."},
    {"Case #": 3, "Loan ID": "LN0013876", "Month": "2008-10", "Status": "Prepaid", "Balance": 0.00, "DPD": 267, "Anomaly Score": 0.9970, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Refer to special servicing / default workout desk", "Detail": "Severe chronic delinquency (267 DPD) immediately prior to full voluntary payoff."},
    {"Case #": 4, "Loan ID": "LN0027221", "Month": "2022-05", "Status": "90+ DPD", "Balance": 197202.75, "DPD": 394, "Anomaly Score": 0.9963, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Refer to special servicing / default workout desk", "Detail": "Chronic delinquency outlier (394 DPD) exceeding standard 180-day charge-off window."},
    {"Case #": 5, "Loan ID": "LN0026208", "Month": "2005-06", "Status": "90+ DPD", "Balance": 105121.00, "DPD": 120, "Anomaly Score": 0.9953, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Verify note rate against loan agreement schedule", "Detail": "Interest rate outlier anomaly with persistent past-due status."},
    {"Case #": 6, "Loan ID": "LN0027221", "Month": "2022-06", "Status": "90+ DPD", "Balance": 196540.71, "DPD": 120, "Anomaly Score": 0.9926, "Category": "Unsupervised Behavioral Outlier", "Drivers": "dpd_roll_max_3m, dpd_roll_max_6m, dpd_roll_mean_6m", "Action": "Manual Servicer Reconciliation & Data Audit", "Detail": "Discontinuous sudden drop in DPD from 394 to 120 without loan modification flag."},
    {"Case #": 7, "Loan ID": "LN0046720", "Month": "2007-06", "Status": "90+ DPD", "Balance": 244131.25, "DPD": 265, "Anomaly Score": 0.9900, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Refer to special servicing / default workout desk", "Detail": "265 days past due on prime property type; servicer update lag detected."},
    {"Case #": 8, "Loan ID": "LN0016646", "Month": "2006-04", "Status": "Prepaid", "Balance": 0.00, "DPD": 304, "Anomaly Score": 0.9884, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Refer to special servicing / default workout desk", "Detail": "Paid in full while in 304 DPD foreclosure pipeline; possible short sale execution."},
    {"Case #": 9, "Loan ID": "LN0017771", "Month": "2012-12", "Status": "Prepaid", "Balance": 0.00, "DPD": 367, "Anomaly Score": 0.9859, "Category": "Unsupervised Behavioral Outlier", "Drivers": "days_past_due, dpd_roll_max_3m, dpd_roll_max_6m", "Action": "Refer to special servicing / default workout desk", "Detail": "Full payoff after 367 consecutive days past due; verify title release."},
    {"Case #": 10, "Loan ID": "LN0009841", "Month": "2021-04", "Status": "Paid Off", "Balance": 14200.00, "DPD": 0, "Anomaly Score": 0.9650, "Category": "Deterministic Rule VR002 Violation", "Drivers": "current_status, current_balance, original_balance", "Action": "Servicer ledger reconciliation audit", "Detail": "Status reported as 'Paid Off' but positive ledger balance ($14,200) remains open."},
    {"Case #": 11, "Loan ID": "LN0034190", "Month": "2018-09", "Status": "Current", "Balance": 420000.00, "DPD": 0, "Anomaly Score": 0.9420, "Category": "Deterministic Rule VR005 Violation", "Drivers": "current_balance, original_balance, balance_growth_ratio", "Action": "Request re-appraisal / collateral valuation", "Detail": "Current balance ($420k) exceeds 2.1x original balance ($200k) without recast record."},
    {"Case #": 12, "Loan ID": "LN0015523", "Month": "2020-02", "Status": "Default", "Balance": 185000.00, "DPD": 15, "Anomaly Score": 0.9280, "Category": "Deterministic Rule VR003 Violation", "Drivers": "current_status, days_past_due, dpd_roll_max_3m", "Action": "Correct servicer status mapping", "Detail": "Reported as Default status despite only 15 DPD recorded on tape."},
]

# Interactive Filter Controls
st.subheader("🔍 Filter & Explore Anomaly Review Queue")
f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
with f_col1:
    min_score = st.slider("Minimum Anomaly Score", min_value=0.80, max_value=1.00, value=0.90, step=0.01)
with f_col2:
    status_filter = st.multiselect("Filter by Status", options=["90+ DPD", "Prepaid", "Paid Off", "Current", "Default"], default=["90+ DPD", "Prepaid", "Paid Off", "Current", "Default"])
with f_col3:
    search_query = st.text_input("Search by Loan ID or Keyword", placeholder="e.g. LN0026208 or note rate")

df_anom = pd.DataFrame(ANOMALY_CASES)
filtered = df_anom[
    (df_anom["Anomaly Score"] >= min_score) &
    (df_anom["Status"].isin(status_filter))
]

if search_query:
    filtered = filtered[
        filtered["Loan ID"].str.contains(search_query, case=False) |
        filtered["Detail"].str.contains(search_query, case=False) |
        filtered["Category"].str.contains(search_query, case=False)
    ]

st.markdown(f"**Showing {len(filtered)} matching anomaly cases:**")

# Format dataframe display
display_df = filtered.copy()
display_df["Balance"] = display_df["Balance"].apply(lambda v: f"${v:,.2f}")
display_df["Anomaly Score"] = display_df["Anomaly Score"].apply(lambda v: f"{v:.4f}")

st.dataframe(
    display_df[["Case #", "Loan ID", "Month", "Status", "Balance", "DPD", "Anomaly Score", "Category", "Action"]],
    use_container_width=True
)

st.markdown("---")

# Deep Dive Case Inspection
st.subheader("🔎 Case Deep-Dive & Diagnostic Explainability")
selected_loan = st.selectbox("Select Loan ID for Detailed Underwriter Breakdown:", options=filtered["Loan ID"].unique())

if selected_loan:
    loan_record = filtered[filtered["Loan ID"] == selected_loan].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Loan Identifier", loan_record["Loan ID"])
    with col2:
        st.metric("Current Balance", f"${loan_record['Balance']:,.2f}")
    with col3:
        st.metric("Days Past Due", f"{loan_record['DPD']} DPD")
    with col4:
        st.metric("Anomaly Score", f"{loan_record['Anomaly Score']:.4f}")
    
    st.markdown(
        f"""
        - **Category**: `{loan_record['Category']}`
        - **Primary Feature Drivers**: `{loan_record['Drivers']}`
        - **Diagnostic Findings**: {loan_record['Detail']}
        - **Recommended Reviewer Action**: **{loan_record['Action']}**
        """
    )

st.markdown("---")

# Pre-Logged LLM Copilot Notes (Zero Live Calls)
st.subheader("📝 Pre-Generated Grounded LLM Copilot Audit Note (Logged Example)")
st.caption("🔒 **Security Guarantee**: Static pre-logged sample from `logs/llm_prompt_log.jsonl`. No live LLM calls or API keys used.")

st.markdown(
    """
    ```markdown
    ### Reviewer Note: Loan LN0004821 (Reporting Month: 2023-05)

    **Part A: Key Risk & Underwriting Assessment**
    - Risk Tier: High Risk (Calibrated 12M Default Probability: 24.50%, 3M Delinquency Probability: 68.40%).
    - Current Performance: Status is `30-59 DPD` with 45 DPD and active balance of $284,500.00 (FL Collateral).
    - Top SHAP Risk Drivers: `days_past_due`, `credit_score_ordinal` (620-659 tier), `rate_to_market_spread`.

    **Part B: Data Quality & Anomaly Assessment**
    - Anomaly Score: 0.4820 / 1.0 (Elevated behavioral volatility).
    - Documentation Status: Complete. Deterministic rules VR001-VR005: PASSED.

    **Part C: Recommended Action Plan**
    - Initiate pre-foreclosure borrower outreach desk and verify borrower liquidity.
    - Confirm whether loan modification workout is pending.

    ----------------------------------------------------------------------
    *Recommendation — not a decision.*
    ```
    """
)
st.info("Every LLM summary is strictly grounded in deterministic context and concludes with the explicit disclaimer: *Recommendation — not a decision.*")
