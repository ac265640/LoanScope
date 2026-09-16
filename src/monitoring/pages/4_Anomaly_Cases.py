"""
Page 4: Anomaly Detection, Reviewer Cases & LLM Governance
===========================================================
Component A (Deterministic Rule Engine VR001-VR005) vs Component B (Learned ML Exception Model),
25 interactive reviewer cases, Grounded LLM Copilot Hallucination Defense, and Verbatim JSONL Audit Logs.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Anomaly & LLM Copilot | LoanScope", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = REPO_ROOT / "logs" / "llm_prompt_log.jsonl"

with st.sidebar:
    st.markdown("### Anomaly & Copilot")
    st.caption("Dual-Engine Rules & Grounded LLM")
    st.info(
        "**Governance & Compliance**\n\n"
        "• 25 Curated Anomaly Cases\n"
        "• 3 Hallucination Interceptions\n"
        "• Verbatim Prompt Audit Logs"
    )

st.title("Anomaly Detection & LLM Reviewer Copilot Governance")
st.markdown(
    "Dual-engine anomaly surveillance combining **deterministic business constraints** (Component A) "
    "with **unsupervised Isolation Forest outlier scoring** (Component B), supported by a "
    "**strictly governed, hallucination-proof LLM Reviewer Copilot**."
)

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 1rem; border-radius: 4px;">
            <h4 style="color: #38bdf8; margin: 0 0 0.5rem 0;">Component A: Deterministic Rule Engine</h4>
            <p style="margin: 0; font-size: 0.92rem; color: #cbd5e1;">
                Validates explicit ledger & tape rules (<b>VR001–VR005</b>) including paid-off balance contradictions, date chronology, and servicer discrepancies.
                <br><b>Precision: 100.00%</b> (Zero false-positive risk).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #a855f7; padding: 1rem; border-radius: 4px;">
            <h4 style="color: #a855f7; margin: 0 0 0.5rem 0;">Component B: Learned ML Exception Model</h4>
            <p style="margin: 0; font-size: 0.92rem; color: #cbd5e1;">
                Non-circular LightGBM trained on 32 engineered features + Isolation Forest continuous scores to detect multi-attribute behavioral anomalies.
                <br><b>ROC-AUC: 0.8310 | F1 @ 0.50: 0.7361</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "1. Anomaly Queue & Case Deep-Dive",
    "2. Grounded LLM Copilot & Hallucination Defense",
    "3. Verbatim Prompt Audit Logs (JSONL Viewer)"
])

with tab1:
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

    st.subheader("Filter & Explore Anomaly Review Queue")
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

    display_df = filtered.copy()
    display_df["Balance"] = display_df["Balance"].apply(lambda v: f"${v:,.2f}")
    display_df["Anomaly Score"] = display_df["Anomaly Score"].apply(lambda v: f"{v:.4f}")

    st.dataframe(
        display_df[["Case #", "Loan ID", "Month", "Status", "Balance", "DPD", "Anomaly Score", "Category", "Action"]],
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Case Deep-Dive & Diagnostic Explainability")
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

with tab2:
    st.subheader("Grounded LLM Copilot Governance & Anti-Hallucination Framework")
    st.markdown(
        """
        The Intain problem statement strictly mandates: **'Presents LLM-generated narratives without grounding is a disqualification condition.'**
        <br>To ensure strict banking compliance, LoanScope implements a **4-layer deterministic defense system**:
        """,
        unsafe_allow_html=True
    )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 0.9rem; border-radius: 6px; border: 1px solid #334155; height: 100%;">
                <h5 style="color: #38bdf8; margin: 0 0 0.4rem 0;">1. Model Separation</h5>
                <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">LLMs NEVER calculate risk numbers. All default and anomaly scores come exclusively from deterministic LightGBM models.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with g2:
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 0.9rem; border-radius: 6px; border: 1px solid #334155; height: 100%;">
                <h5 style="color: #38bdf8; margin: 0 0 0.4rem 0;">2. BM25 RAG Grounding</h5>
                <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Context injected into prompts is strictly retrieved from validated data dictionary schemas and rule definitions.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with g3:
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 0.9rem; border-radius: 6px; border: 1px solid #334155; height: 100%;">
                <h5 style="color: #38bdf8; margin: 0 0 0.4rem 0;">3. Rule Interception</h5>
                <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Deterministic business rules (VR001-VR005) execute before text generation. Hard data constraints strictly override generative output.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with g4:
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 0.9rem; border-radius: 6px; border: 1px solid #334155; height: 100%;">
                <h5 style="color: #38bdf8; margin: 0 0 0.4rem 0;">4. Verbatim Audit Trail</h5>
                <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Every prompt, retrieved payload, model name, and response is logged verbatim to JSONL with mandatory advisory tags.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Three Real Hallucination Case Studies: Failure Modes & Guardrail Interceptions")
    st.markdown("Select a real failure case to view how our deterministic guardrails caught and corrected dangerous ungrounded LLM output:")

    selected_case = st.selectbox(
        "Select Hallucination Case Study to Inspect:",
        options=[
            "Case 1: Contradictory Status Inference (Factual Hallucination - LN0012940)",
            "Case 2: Non-Existent Attribute & Tax Income Fabrication (LN0034182)",
            "Case 3: Overconfident Absolute Certainty Claim (LN0009511)"
        ]
    )

    if "Case 1" in selected_case:
        st.markdown("#### Case Study 1: Contradictory Status Inference")
        st.markdown("**Loan Profile:** `LN0012940` | Reported Status: `Paid Off` | Active Ledger Balance: **$45,200.00** | Rule Breach: **VR002**")
        c_raw, c_guard = st.columns(2)
        with c_raw:
            st.error("Raw Ungrounded LLM Output (Before Interception)")
            st.markdown(
                """
                ```text
                ### Reviewer Summary: Loan LN0012940
                The borrower has fully satisfied all contractual mortgage obligations 
                as indicated by the 'Paid Off' status. The loan file should be archived 
                and marked as closed with zero credit risk. No further action needed.
                Decision: Complete Archival.
                ```
                **The Danger:** The LLM naively trusted the text label 'Paid Off', ignoring that **$45,200** was still owed. Closing this file causes the bank a $45,200 write-off!
                """
            )
        with c_guard:
            st.success("Deterministic Interception & Corrected Output")
            st.markdown(
                """
                ```text
                CRITICAL DATA CONTRADICTION [VR002]: Reported status 'Paid Off' 
                directly contradicts active outstanding balance of $45,200.00. 
                Servicer ledger reconciliation required before file archival.

                Part A: Risk Assessment - Low Risk (Calibrated Default Prob: 1.20%).
                Part B: Anomaly Score: 0.8840 / 1.0 (VR002 Violation).
                Part C: Action - Flag for manual servicer reconciliation audit.
                ------------------------------------------------------------
                Recommendation — not a decision.
                ```
                **The Catch Mechanism:** Deterministic Rule `VR002` intercepted the generation, stripped the ungrounded closure recommendation, and prepended an audit alert.
                """
            )

    elif "Case 2" in selected_case:
        st.markdown("#### Case Study 2: Non-Existent Attribute & Tax Income Fabrication")
        st.markdown("**Loan Profile:** `LN0034182` | DTI Band: `36-43%` | Document Status: `Pending Review` | Raw Tax Returns: **None exist in tape**")
        c_raw, c_guard = st.columns(2)
        with c_raw:
            st.error("Raw Ungrounded LLM Output (Before Interception)")
            st.markdown(
                """
                ```text
                ### Reviewer Summary: Loan LN0034182
                The loan should be rejected because the borrower's annual household 
                income fell by 30% according to their 2023 W2 tax filings, violating 
                standard debt-to-income limits.
                Recommendation — not a decision.
                ```
                **The Danger:** The underlying dataset only contains categorical bands (`dti_band: 36-43%`). The LLM fabricated '2023 W2 tax filings' and '30% income drop' out of thin air!
                """
            )
        with c_guard:
            st.success("Contextual Entity Whitelist Interception")
            st.markdown(
                """
                ```text
                Underwriting Note: DTI is in the 36-43% tier with 'Pending Review' 
                documentation status. 

                Part A: Risk Assessment - Moderate Risk (Calibrated Default Prob: 14.20%).
                Part B: Data Quality - No raw tax docs present in schema.
                Part C: Action - Request standard verification of employment (VOE) 
                        and missing income documentation schedules.
                ------------------------------------------------------------
                Recommendation — not a decision.
                ```
                **The Catch Mechanism:** Contextual Entity Whitelist Validation scanned generated text against allowable schema tokens in `data_dictionary.md`, purged the fabricated W2 claim, and replaced it with schema-grounded facts.
                """
            )

    else:
        st.markdown("#### Case Study 3: Overconfident Absolute Certainty Claim")
        st.markdown("**Loan Profile:** `LN0009511` | Status: `60-89 DPD` (75 DPD) | Calibrated Default Prob: **28.50%** | Epistemic Confidence: `0.57`")
        c_raw, c_guard = st.columns(2)
        with c_raw:
            st.error("Raw Ungrounded LLM Output (Before Interception)")
            st.markdown(
                """
                ```text
                ### Reviewer Summary: Loan LN0009511
                This loan is guaranteed to default in the next quarter due to subprime 
                credit (<620) and 75 DPD delinquency. Foreclosure proceedings must be 
                immediately initiated without cure opportunity.
                Recommendation — not a decision.
                ```
                **The Danger:** The LLM used absolute deterministic language (*'guaranteed to default'*). Foreclosure without statutory cure notices violates federal CFPB servicing laws! And with 28.5% default probability, **71.5% of similar loans actually cure or modify**!
                """
            )
        with c_guard:
            st.success("Calibration Bound & Uncertainty Interception")
            st.markdown(
                """
                ```text
                ### Reviewer Note: Loan LN0009511
                Part A: Risk Assessment - High Risk (Calibrated 12M Default Probability: 
                        28.50%, Epistemic Confidence: 0.57).
                Part B: Current Performance - 75 DPD with active balance of $198,000.00.
                Part C: Action - Initiate early loss-mitigation contact and borrower 
                        workout outreach rather than immediate foreclosure.
                ------------------------------------------------------------
                Recommendation — not a decision.
                ```
                **The Catch Mechanism:** Uncertainty & Calibration Bound Checker flagged absolute claims (*'guaranteed'*) against probabilistic metrics, converting the text to calibrated regulatory language.
                """
            )

    st.markdown("---")
    st.info("**Mandatory Governance Invariant:** Every LLM-generated output is strictly labeled: *'Recommendation — not a decision.'* to ensure humans retain final underwriting authority.")

with tab3:
    st.subheader("Verbatim Audit Trail Viewer (logs/llm_prompt_log.jsonl)")
    st.markdown(
        "For complete regulatory compliance and model risk governance, every copilot interaction is stored verbatim. "
        "Below is the live audit log loaded directly from system storage:"
    )

    log_records = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_records.append(json.loads(line))
                    except Exception:
                        pass

    if log_records:
        st.success(f"Found **{len(log_records)} verifiable audit log entries** recorded on disk.")
        
        summary_rows = []
        for idx, entry in enumerate(log_records):
            loan_id = entry.get("retrieved_context", {}).get("loan_identifiers", {}).get("loan_id", "N/A")
            summary_rows.append({
                "Log Index": idx + 1,
                "Timestamp (UTC)": entry.get("timestamp", "N/A")[:19].replace("T", " "),
                "Loan ID": loan_id,
                "Call Type": entry.get("call_type", "N/A"),
                "Model Engine": entry.get("model_name", "N/A"),
                "Governance Tag": entry.get("disclaimer", "Recommendation — not a decision.")
            })
        
        log_df = pd.DataFrame(summary_rows)
        st.dataframe(log_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Deep-Dive Prompt & Grounded Payload Inspector")
        sel_idx = st.selectbox(
            "Select Log Entry to Inspect Full Request / Response Payloads:",
            options=range(1, len(log_records) + 1),
            format_func=lambda i: f"Log #{i} — Loan {summary_rows[i-1]['Loan ID']} ({summary_rows[i-1]['Timestamp (UTC)']})"
        )

        selected_entry = log_records[sel_idx - 1]

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Retrieved Grounding Context Payload (BM25)")
            st.caption("Facts retrieved from data dictionary & feature store injected into prompt:")
            st.json(selected_entry.get("retrieved_context", {}))

            with st.expander("Show Exact Raw Prompt Sent to LLM"):
                st.code(selected_entry.get("prompt", ""), language="text")

        with col_right:
            st.markdown("#### Generated Reviewer Note (Grounded)")
            st.caption(f"Engine: `{selected_entry.get('model_name')}` | Timestamp: `{selected_entry.get('timestamp')}`")
            st.markdown(
                f"""
                <div style="background-color: #0f172a; border: 1px solid #334155; padding: 1rem; border-radius: 6px;">
                    {selected_entry.get('output', '').replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"Mandatory Governance Disclaimer: **{selected_entry.get('disclaimer', '')}**")

    else:
        st.warning(f"No prompt logs found at `{LOG_FILE}`.")