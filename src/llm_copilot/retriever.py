"""
Context Retrieval Module for Grounded LLM Reviewer Copilot
==========================================================
Retrieves pertinent field definitions from data/data_dictionary.md, active rule definitions
from data/validation_rules.json, and structured loan metrics for context-augmented generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


class GroundedContextRetriever:
    """Retrieves authoritative reference metadata and tabular loan facts."""

    def __init__(self):
        self.dict_path = DATA_DIR / "data_dictionary.md"
        self.rules_path = DATA_DIR / "validation_rules.json"

        self.data_dict_text = ""
        if self.dict_path.exists():
            with open(self.dict_path) as f:
                self.data_dict_text = f.read()

        self.validation_rules = []
        if self.rules_path.exists():
            with open(self.rules_path) as f:
                r_json = json.load(f)
                self.validation_rules = r_json.get("rules", [])

    def retrieve_context_for_loan(
        self,
        loan_record: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        anomalies: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Construct a structured, complete grounding payload for LLM injection."""
        # 1. Extract active field values
        loan_id = loan_record.get("loan_id", "UNKNOWN")
        status = loan_record.get("current_status", "Unknown")
        bal = loan_record.get("current_balance", 0.0)
        orig_bal = loan_record.get("original_balance", 0.0)
        rate = loan_record.get("interest_rate", 0.0)
        dpd = loan_record.get("days_past_due", 0)
        credit_band = loan_record.get("credit_score_band", "MISSING")
        ltv = loan_record.get("ltv_band", "Unknown")
        dti = loan_record.get("dti_band", "Unknown")
        state = loan_record.get("state", "Unknown")
        doc_status = loan_record.get("document_status", "Complete")
        rep_month = loan_record.get("reporting_month", "Unknown")
        orig_month = loan_record.get("origination_month", "Unknown")

        # 2. Match relevant deterministic validation rules
        applicable_rules = []
        for r in self.validation_rules:
            applicable_rules.append({
                "rule_id": r["rule_id"],
                "rule_name": r["name"],
                "condition": r["condition"],
                "severity": r["severity"],
            })

        # 3. Compile ground-truth context dict
        context = {
            "loan_identifiers": {
                "loan_id": loan_id,
                "reporting_month": rep_month,
                "origination_month": orig_month,
            },
            "financial_profile": {
                "current_balance_usd": bal,
                "original_balance_usd": orig_bal,
                "note_interest_rate_pct": rate,
                "credit_score_tier": credit_band,
                "ltv_tier": ltv,
                "dti_tier": dti,
                "collateral_state": state,
                "documentation_status": doc_status,
            },
            "servicing_and_delinquency": {
                "current_performance_status": status,
                "days_past_due": dpd,
                "modification_flag": loan_record.get("modification_flag", 0),
            },
            "ml_model_outputs": {
                "prob_next_3m_delinquency": ml_predictions.get("prob_next_3m_delinquency", 0.0),
                "prob_next_6m_delinquency": ml_predictions.get("prob_next_6m_delinquency", 0.0),
                "prob_next_12m_default": ml_predictions.get("prob_next_12m_default", 0.0),
                "prob_next_12m_prepayment": ml_predictions.get("prob_next_12m_prepayment", 0.0),
                "predicted_next_state": ml_predictions.get("next_state", status),
                "anomaly_score": ml_predictions.get("anomaly_score", 0.0),
                "top_shap_drivers": ml_predictions.get("top_drivers", []),
                "confidence_score": ml_predictions.get("confidence", 0.85),
            },
            "active_validation_rules": applicable_rules,
            "governance_disclaimer": "CRITICAL: All model outputs are advisory. Output must be formulated as a 'Recommendation — not a decision.'",
        }

        return context
