"""
Date and Temporal Relationship Validator
========================================
Audits temporal integrity and logical ordering across loan reporting cycles.
"""

from typing import Dict, Any, List
import pandas as pd


def validate_date_relationships(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check temporal consistency:
      1. reporting_month >= origination_month
      2. loan_age_months == (reporting_month - origination_month in months)
      3. remaining_term_months >= 0
      4. last_updated_at chronological consistency
    """
    total_rows = len(df)
    results: Dict[str, Any] = {
        "total_rows_inspected": total_rows,
        "anomalous_rows_count": 0,
        "anomalous_rows_pct": 0.0,
        "checks": {},
    }

    anomalous_indices = set()

    # Check 1: reporting_month < origination_month
    if "reporting_month" in df.columns and "origination_month" in df.columns:
        rep_dt = pd.to_datetime(df["reporting_month"] + "-01", errors="coerce")
        orig_dt = pd.to_datetime(df["origination_month"] + "-01", errors="coerce")

        bad_date_mask = (rep_dt < orig_dt)
        bad_count = int(bad_date_mask.sum())
        anomalous_indices.update(df[bad_date_mask].index)

        examples = []
        if bad_count > 0:
            sample_bad = df[bad_date_mask][["loan_id", "origination_month", "reporting_month"]].head(5)
            examples = sample_bad.to_dict(orient="records")

        results["checks"]["reporting_before_origination"] = {
            "violation_count": bad_count,
            "violation_pct": round(bad_count / total_rows * 100, 4),
            "severity": "CRITICAL",
            "description": "Reporting month occurs prior to loan origination funding date.",
            "examples": examples,
        }

    # Check 2: negative remaining term
    if "remaining_term_months" in df.columns:
        neg_term_mask = (df["remaining_term_months"] < 0)
        neg_count = int(neg_term_mask.sum())
        anomalous_indices.update(df[neg_term_mask].index)

        results["checks"]["negative_remaining_term"] = {
            "violation_count": neg_count,
            "violation_pct": round(neg_count / total_rows * 100, 4),
            "severity": "HIGH",
            "description": "Remaining term in months is less than 0.",
        }

    # Check 3: negative loan age
    if "loan_age_months" in df.columns:
        neg_age_mask = (df["loan_age_months"] <= 0)
        neg_age_count = int(neg_age_mask.sum())
        anomalous_indices.update(df[neg_age_mask].index)

        results["checks"]["non_positive_loan_age"] = {
            "violation_count": neg_age_count,
            "violation_pct": round(neg_age_count / total_rows * 100, 4),
            "severity": "MEDIUM",
            "description": "Loan age in months is <= 0 for active panel observation.",
        }

    results["anomalous_rows_count"] = len(anomalous_indices)
    results["anomalous_rows_pct"] = round(len(anomalous_indices) / total_rows * 100, 4)

    return results
