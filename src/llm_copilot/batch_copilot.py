"""
Batch Grounded Reviewer Copilot
================================
Runs the grounded LLM copilot across a diverse portfolio of loan profiles
representing different risk tiers, performance states, and exception types.
Each invocation is logged verbatim to logs/llm_prompt_log.jsonl.
"""

import logging
from pathlib import Path

from src.llm_copilot.copilot import GroundedReviewerCopilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Representative loan profiles covering the full risk spectrum
BATCH_LOANS = [
    # 1. High-Risk — deep delinquency
    {
        "loan": {
            "loan_id": "LN0004821", "reporting_month": "2023-05", "origination_month": "2019-03",
            "current_status": "30-59 DPD", "days_past_due": 45, "current_balance": 284500.0,
            "original_balance": 310000.0, "interest_rate": 5.875, "credit_score_band": "620-659",
            "ltv_band": "80-90%", "dti_band": "36-43%", "state": "FL",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.684, "prob_next_6m_delinquency": 0.742,
            "prob_next_12m_default": 0.245, "prob_next_12m_prepayment": 0.031,
            "next_state": "60-89 DPD", "anomaly_score": 0.482,
            "top_drivers": ["days_past_due", "credit_score_ordinal", "rate_to_market_spread"],
            "confidence": 0.88,
        },
    },
    # 2. Imminent Default — severely delinquent subprime
    {
        "loan": {
            "loan_id": "LN0009511", "reporting_month": "2023-08", "origination_month": "2018-06",
            "current_status": "60-89 DPD", "days_past_due": 75, "current_balance": 198000.0,
            "original_balance": 225000.0, "interest_rate": 7.250, "credit_score_band": "<620",
            "ltv_band": "90-100%", "dti_band": ">43%", "state": "TX",
            "document_status": "Missing Items", "modification_flag": 1,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.921, "prob_next_6m_delinquency": 0.888,
            "prob_next_12m_default": 0.571, "prob_next_12m_prepayment": 0.012,
            "next_state": "Default", "anomaly_score": 0.731,
            "top_drivers": ["days_past_due", "dti_band_ordinal", "credit_score_ordinal"],
            "confidence": 0.57,
        },
    },
    # 3. Likely Prepayment — prime borrower, rate incentive
    {
        "loan": {
            "loan_id": "LN0022311", "reporting_month": "2023-04", "origination_month": "2021-11",
            "current_status": "Current", "days_past_due": 0, "current_balance": 412000.0,
            "original_balance": 450000.0, "interest_rate": 6.875, "credit_score_band": "740+",
            "ltv_band": "<60%", "dti_band": "<28%", "state": "CA",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.018, "prob_next_6m_delinquency": 0.031,
            "prob_next_12m_default": 0.007, "prob_next_12m_prepayment": 0.312,
            "next_state": "Current", "anomaly_score": 0.143,
            "top_drivers": ["rate_to_market_spread", "balance_to_orig_ratio", "loan_age_months"],
            "confidence": 0.94,
        },
    },
    # 4. Data Conflict — Paid Off but positive balance
    {
        "loan": {
            "loan_id": "LN0012940", "reporting_month": "2023-06", "origination_month": "2017-04",
            "current_status": "Paid Off", "days_past_due": 0, "current_balance": 45200.0,
            "original_balance": 185000.0, "interest_rate": 4.125, "credit_score_band": "700-739",
            "ltv_band": "60-80%", "dti_band": "28-35%", "state": "NY",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.041, "prob_next_6m_delinquency": 0.052,
            "prob_next_12m_default": 0.012, "prob_next_12m_prepayment": 0.088,
            "next_state": "Current", "anomaly_score": 0.884,
            "top_drivers": ["balance_change_1m_pct", "balance_to_orig_ratio", "interest_rate_imputed"],
            "confidence": 0.71,
        },
    },
    # 5. Watch-List — moderate delinquency, modified loan
    {
        "loan": {
            "loan_id": "LN0031702", "reporting_month": "2023-07", "origination_month": "2020-02",
            "current_status": "30-59 DPD", "days_past_due": 32, "current_balance": 310800.0,
            "original_balance": 325000.0, "interest_rate": 5.250, "credit_score_band": "660-699",
            "ltv_band": "80-90%", "dti_band": "36-43%", "state": "IL",
            "document_status": "Pending Review", "modification_flag": 1,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.521, "prob_next_6m_delinquency": 0.489,
            "prob_next_12m_default": 0.142, "prob_next_12m_prepayment": 0.048,
            "next_state": "60-89 DPD", "anomaly_score": 0.562,
            "top_drivers": ["rolling_delinquent_3m", "modification_flag", "days_past_due"],
            "confidence": 0.76,
        },
    },
    # 6. Low-Risk Performing — vanilla current loan
    {
        "loan": {
            "loan_id": "LN0041200", "reporting_month": "2023-09", "origination_month": "2022-03",
            "current_status": "Current", "days_past_due": 0, "current_balance": 287600.0,
            "original_balance": 300000.0, "interest_rate": 6.125, "credit_score_band": "720-739",
            "ltv_band": "80-90%", "dti_band": "28-35%", "state": "GA",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.032, "prob_next_6m_delinquency": 0.054,
            "prob_next_12m_default": 0.018, "prob_next_12m_prepayment": 0.091,
            "next_state": "Current", "anomaly_score": 0.211,
            "top_drivers": ["credit_score_ordinal", "loan_age_months", "balance_to_orig_ratio"],
            "confidence": 0.92,
        },
    },
    # 7. Vintage stress — pre-2010 origination, MNAR credit
    {
        "loan": {
            "loan_id": "LN0007834", "reporting_month": "2023-10", "origination_month": "2007-08",
            "current_status": "Current", "days_past_due": 0, "current_balance": 52400.0,
            "original_balance": 320000.0, "interest_rate": 6.750, "credit_score_band": "620-659",
            "ltv_band": "60-80%", "dti_band": "28-35%", "state": "MI",
            "document_status": "Complete", "modification_flag": 1,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.112, "prob_next_6m_delinquency": 0.198,
            "prob_next_12m_default": 0.071, "prob_next_12m_prepayment": 0.182,
            "next_state": "Current", "anomaly_score": 0.391,
            "top_drivers": ["loan_age_months", "balance_to_orig_ratio", "rolling_delinquent_6m"],
            "confidence": 0.83,
        },
    },
    # 8. High-DTI borderline — intervention candidate
    {
        "loan": {
            "loan_id": "LN0055012", "reporting_month": "2023-11", "origination_month": "2021-05",
            "current_status": "Current", "days_past_due": 0, "current_balance": 398200.0,
            "original_balance": 410000.0, "interest_rate": 6.500, "credit_score_band": "660-699",
            "ltv_band": "80-90%", "dti_band": ">43%", "state": "AZ",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.187, "prob_next_6m_delinquency": 0.241,
            "prob_next_12m_default": 0.098, "prob_next_12m_prepayment": 0.063,
            "next_state": "30-59 DPD", "anomaly_score": 0.334,
            "top_drivers": ["dti_band_ordinal", "credit_score_ordinal", "rate_to_market_spread"],
            "confidence": 0.79,
        },
    },
    # 9. Balance growth outlier — possible forbearance or capitalization
    {
        "loan": {
            "loan_id": "LN0038741", "reporting_month": "2023-12", "origination_month": "2019-09",
            "current_status": "30-59 DPD", "days_past_due": 38, "current_balance": 499800.0,
            "original_balance": 350000.0, "interest_rate": 4.875, "credit_score_band": "700-739",
            "ltv_band": ">100%", "dti_band": "36-43%", "state": "NV",
            "document_status": "Pending Review", "modification_flag": 1,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.443, "prob_next_6m_delinquency": 0.502,
            "prob_next_12m_default": 0.168, "prob_next_12m_prepayment": 0.022,
            "next_state": "60-89 DPD", "anomaly_score": 0.791,
            "top_drivers": ["balance_change_1m_pct", "balance_to_orig_ratio", "days_past_due"],
            "confidence": 0.68,
        },
    },
    # 10. Recently recovered — was delinquent, back to current
    {
        "loan": {
            "loan_id": "LN0018823", "reporting_month": "2023-09", "origination_month": "2020-08",
            "current_status": "Current", "days_past_due": 0, "current_balance": 267500.0,
            "original_balance": 280000.0, "interest_rate": 3.750, "credit_score_band": "680-699",
            "ltv_band": "80-90%", "dti_band": "28-35%", "state": "WA",
            "document_status": "Complete", "modification_flag": 0,
        },
        "preds": {
            "prob_next_3m_delinquency": 0.231, "prob_next_6m_delinquency": 0.198,
            "prob_next_12m_default": 0.059, "prob_next_12m_prepayment": 0.148,
            "next_state": "Current", "anomaly_score": 0.289,
            "top_drivers": ["rolling_delinquent_3m", "dpd_rolling_max_6m", "credit_score_ordinal"],
            "confidence": 0.81,
        },
    },
]


def run_batch_copilot():
    """Run the grounded reviewer copilot on all 10 diverse loan profiles."""
    log.info(f"Starting batch copilot run across {len(BATCH_LOANS)} loan profiles...")
    copilot = GroundedReviewerCopilot()

    results = []
    for i, item in enumerate(BATCH_LOANS, 1):
        loan = item["loan"]
        preds = item["preds"]
        log.info(f"[{i}/{len(BATCH_LOANS)}] Generating reviewer note for loan {loan['loan_id']}...")
        result = copilot.generate_reviewer_note(loan, preds)
        results.append(result)
        print(f"\n{'='*60}")
        print(f"Loan {loan['loan_id']} | Risk status: {loan['current_status']} | Model: {result['model_name']}")
        print(f"{'='*60}")
        print(result["reviewer_note"])

    log.info(f"✅ Batch copilot complete. {len(results)} reviewer notes logged to logs/llm_prompt_log.jsonl")
    return results


if __name__ == "__main__":
    run_batch_copilot()
