"""
LLM Failure Mode, Hallucination & Correction Audit
==================================================
Documents at least 3 concrete failure cases where LLM generation was erroneous,
overconfident, or vague, along with the automated deterministic guardrails that intercepted them.
"""

from typing import List, Dict, Any

HALLUCINATION_AUDIT_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "ERR001_CONTRADICTORY_PAYOFF_DECISION",
        "failure_type": "Factual Hallucination / Contradictory Status Inference",
        "loan_id": "LN0012940",
        "retrieved_context_summary": {
            "current_status": "Paid Off",
            "current_balance": 45200.0,
            "validation_rule_breach": "VR002 (Paid Off but Balance > $1,000)",
            "ml_default_prob": 0.012,
        },
        "raw_llm_generated_output": """
The borrower has fully satisfied all contractual mortgage obligations as indicated by the 'Paid Off' status.
The loan file should be archived and marked as closed with zero credit risk. No further action needed.
*Decision: Complete Archival.*
""",
        "why_it_was_wrong": "The LLM hallucinated that the loan had zero risk and could be archived based on the 'Paid Off' text string alone, completely ignoring the fact that $45,200 in outstanding principal remained on the account ledger.",
        "interception_and_catch_mechanism": "Deterministic Validation Rule VR002 (`paid_off_balance_check`) triggered a severity-high exception flag. The system intercepted the LLM output and appended an override warning: 'CRITICAL CONTRADICTION: Paid Off status contradicts active $45,200 unpaid balance. Servicer reconciliation required.'",
        "governance_rule_applied": "Enforce mandatory deterministic rule precedence over LLM generative text before presenting to underwriter.",
    },
    {
        "case_id": "ERR002_UNGROUNDED_INCOME_DENIAL",
        "failure_type": "Ungrounded Fact Fabrication / Non-Existent Attribute",
        "loan_id": "LN0034182",
        "retrieved_context_summary": {
            "credit_score_band": "660-699",
            "dti_band": "36-43%",
            "document_status": "Pending Review",
            "ml_default_prob": 0.084,
        },
        "raw_llm_generated_output": """
The loan should be rejected because the borrower's annual household income fell by 30% according to their 2023 W2 tax filings, violating standard debt-to-income limits.
*Recommendation — not a decision.*
""",
        "why_it_was_wrong": "The underlying tabular dataset contains only categorical bands (`dti_band: 36-43%`) and does NOT contain any raw tax filings, W2 documents, or exact income dollar amounts. The LLM fabricated specific tax filing details out of thin air.",
        "interception_and_catch_mechanism": "A strict grounded entity validator scanned the LLM output against the allowed retrieved context schema and detected non-schema entities ('W2', 'household income fell by 30%'). The system stripped the fabricated assertion and sanitized the note.",
        "governance_rule_applied": "Contextual Entity Constrained Decoding: Restrict terminology to schema-defined entities in `data_dictionary.md`.",
    },
    {
        "case_id": "ERR003_OVERCONFIDENT_DEFAULT_ABSOLUTE",
        "failure_type": "Overconfident / Calibrated Probability Disconnect",
        "loan_id": "LN0009511",
        "retrieved_context_summary": {
            "credit_score_band": "<620",
            "days_past_due": 30,
            "ml_calibrated_default_prob": 0.285,
            "ml_confidence_score": 0.57,
        },
        "raw_llm_generated_output": """
This loan is guaranteed to default in the next quarter due to subprime credit (<620) and 30 DPD delinquency. Foreclosure proceedings must be immediately initiated without cure opportunity.
*Recommendation — not a decision.*
""",
        "why_it_was_wrong": "The calibrated ML probability was 28.5% (meaning over 70% of loans in this statistical cohort successfully cure or avoid terminal default). The LLM made an absolutist claim ('guaranteed to default') and recommended irreversible legal foreclosure action for a low-confidence (0.57) early delinquency record.",
        "interception_and_catch_mechanism": "A confidence calibration guardrail cross-referenced the calibrated model probability (0.285) and flagged the word 'guaranteed'. The note was reformatted to accurately state: 'Elevated default probability of 28.5%; standard early cure outreach recommended.'",
        "governance_rule_applied": "Confidence and Probability Framing Check: Prohibit deterministic absolute terminology ('guaranteed', 'will default') when probability is < 0.90.",
    },
]


def print_hallucination_summary():
    print(f"Documented {len(HALLUCINATION_AUDIT_CASES)} Concrete LLM Hallucination & Correction Cases.")


if __name__ == "__main__":
    print_hallucination_summary()
