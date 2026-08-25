"""
Grounded LLM Reviewer Copilot Engine
====================================
Generates context-grounded reviewer notes and diagnostic summaries using LLM APIs.
Enforces strict grounding, verbatim logging to `logs/llm_prompt_log.jsonl`,
and standard disclaimer: 'Recommendation — not a decision.'
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

from src.llm_copilot.retriever import GroundedContextRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_LOG_PATH = LOGS_DIR / "llm_prompt_log.jsonl"


def log_llm_call(
    prompt: str,
    model_name: str,
    context: Dict[str, Any],
    output: str,
    call_type: str = "reviewer_summary",
):
    """Log every LLM invocation verbatim to jsonl audit log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "call_type": call_type,
        "model_name": model_name,
        "retrieved_context": context,
        "prompt": prompt,
        "output": output,
        "disclaimer": "Recommendation — not a decision.",
    }
    with open(PROMPT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


class GroundedReviewerCopilot:
    """Grounded LLM Reviewer Copilot with multi-backend support (OpenAI / Gemini / Grounded Rule Generator)."""

    def __init__(self):
        self.retriever = GroundedContextRetriever()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")

    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Construct structured prompt with explicit grounding constraints."""
        prompt = f"""You are an expert Credit & Loan Quality Reviewer Copilot assisting an operational credit underwriter.
Your task is to analyze the loan file using ONLY the provided facts and machine learning model predictions.

GROUNDING FACTS & PREDICTIONS:
{json.dumps(context, indent=2)}

INSTRUCTIONS:
1. Provide a concise 3-part reviewer summary:
   - Part A: Key Risk & Underwriting Assessment (synthesize credit band, LTV/DTI, balance trajectory, and predicted default/delinquency probabilities).
   - Part B: Data Quality & Anomaly Flags (highlight any validation rule breaches or anomaly drivers).
   - Part C: Recommended Action Plan (concrete next steps for the loan officer/servicer).
2. NEVER fabricate facts or make assumptions not supported by the context above.
3. Explicitly conclude with the exact text: "Recommendation — not a decision."

REVIEWER NOTE:"""
        return prompt

    def generate_reviewer_note(
        self,
        loan_record: Dict[str, Any],
        ml_predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate grounded reviewer note with complete audit logging."""
        context = self.retriever.retrieve_context_for_loan(loan_record, ml_predictions)
        prompt = self.build_prompt(context)
        model_name = "offline-grounded-engine"
        note_text = ""

        # Attempt Google Gemini
        if self.gemini_key and not self.gemini_key.startswith("your-"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                note_text = response.text
                model_name = "gemini-1.5-flash"
            except Exception as e:
                log.warning(f"Gemini API call failed: {e}. Trying OpenAI fallback...")

        # Attempt OpenAI fallback
        if not note_text and self.openai_key and not self.openai_key.startswith("your-"):
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                note_text = response.choices[0].message.content
                model_name = "gpt-4o-mini"
            except Exception as e:
                log.warning(f"OpenAI API call failed: {e}. Using deterministic grounded generation...")

        # Deterministic Grounded Generation Fallback
        if not note_text:
            model_name = "deterministic-grounded-copilot-v1"
            p_def = context["ml_model_outputs"]["prob_next_12m_default"]
            p_del = context["ml_model_outputs"]["prob_next_3m_delinquency"]
            status = context["servicing_and_delinquency"]["current_performance_status"]
            dpd = context["servicing_and_delinquency"]["days_past_due"]
            bal = context["financial_profile"]["current_balance_usd"]
            drivers = context["ml_model_outputs"]["top_shap_drivers"]

            risk_tier = "High Risk" if p_def >= 0.20 else "Moderate Risk" if p_def >= 0.08 else "Low Risk"

            note_text = f"""### Reviewer Note: Loan {context['loan_identifiers']['loan_id']}

**Part A: Key Risk & Underwriting Assessment**
- **Risk Tier**: {risk_tier} (Calibrated 12M Default Probability: {p_def:.2%}, 3M Delinquency Probability: {p_del:.2%}).
- **Current Performance**: Status is `{status}` with {dpd} DPD and active balance of ${bal:,.2f}.
- **Top Risk Drivers**: {', '.join(drivers) if drivers else 'Standard amortization'}.

**Part B: Data Quality & Anomaly Assessment**
- **Anomaly Score**: {context['ml_model_outputs']['anomaly_score']:.4f} / 1.0.
- **Documentation Status**: `{context['financial_profile']['documentation_status']}`.

**Part C: Recommended Action Plan**
- {'Initiate pre-foreclosure outreach and verify borrower liquidity.' if p_def >= 0.20 else 'Maintain standard surveillance.'}

---
*Recommendation — not a decision.*"""

        # Ensure mandatory disclaimer is present
        if "Recommendation — not a decision." not in note_text:
            note_text += "\n\n*Recommendation — not a decision.*"

        # Log verbatim call
        log_llm_call(prompt, model_name, context, note_text)

        return {
            "loan_id": context["loan_identifiers"]["loan_id"],
            "model_name": model_name,
            "reviewer_note": note_text,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Recommendation — not a decision.",
        }


def main():
    log.info("Running Grounded Reviewer Copilot demo...")
    copilot = GroundedReviewerCopilot()

    sample_loan = {
        "loan_id": "LN0004821",
        "reporting_month": "2023-05",
        "origination_month": "2019-03",
        "current_status": "30-59 DPD",
        "days_past_due": 45,
        "current_balance": 284500.0,
        "original_balance": 310000.0,
        "interest_rate": 5.875,
        "credit_score_band": "620-659",
        "ltv_band": "80-90%",
        "dti_band": "36-43%",
        "state": "FL",
        "document_status": "Complete",
        "modification_flag": 0,
    }

    sample_preds = {
        "prob_next_3m_delinquency": 0.684,
        "prob_next_6m_delinquency": 0.742,
        "prob_next_12m_default": 0.245,
        "prob_next_12m_prepayment": 0.031,
        "next_state": "60-89 DPD",
        "anomaly_score": 0.4820,
        "top_drivers": ["days_past_due", "credit_score_ordinal", "rate_to_market_spread"],
        "confidence": 0.88,
    }

    result = copilot.generate_reviewer_note(sample_loan, sample_preds)
    print("\n" + "=" * 60)
    print(f"Generated Reviewer Note (Model: {result['model_name']}):")
    print("=" * 60)
    print(result["reviewer_note"])
    print("=" * 60)
    log.info(f"✅ Copilot output logged to {PROMPT_LOG_PATH}")


if __name__ == "__main__":
    main()
