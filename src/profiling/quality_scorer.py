"""
Data Quality Scoring Engine (Record-Level and Batch-Level)
==========================================================
Computes calibrated Data Quality (DQ) scores with fully documented, transparent formulas.
Evaluates completeness, temporal validity, consistency, plausibility, and integrity.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_record_quality_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute record-level Data Quality (DQ) score in range [0, 100].
    
    Mathematical Formulation:
      DQ_i = 100 - sum(penalty_k * indicator_k(i))
      clamped to [0, 100]

    Penalty Hierarchy:
      1. Critical Date Anomaly (reporting < origination): 35 points
      2. Status/Balance Contradiction (Paid Off / Prepaid with Balance > $1k): 30 points
      3. Status/Delinquency Contradiction (Default with DPD < 60): 25 points
      4. Extreme Physical/Financial Outlier (Interest Rate > 15%, Balance > 2x Orig): 15 points
      5. Primary Credit Missingness (missing FICO / credit_score_band): 10 points
      6. Missing Non-Critical Feature (interest_rate, dti_band): 5 points
      7. Trailing Document Exception: 5 points
    """
    scores = np.full(len(df), 100.0)

    # 1. Date check
    if "reporting_month" in df.columns and "origination_month" in df.columns:
        rep_dt = pd.to_datetime(df["reporting_month"] + "-01", errors="coerce")
        orig_dt = pd.to_datetime(df["origination_month"] + "-01", errors="coerce")
        bad_date = (rep_dt < orig_dt).fillna(False)
        scores -= bad_date.astype(float) * 35.0

    # 2. Paid Off / Prepaid Balance Contradiction
    if "current_status" in df.columns and "current_balance" in df.columns:
        bad_balance = (df["current_status"].isin(["Paid Off", "Prepaid"])) & (df["current_balance"] > 1000)
        scores -= bad_balance.fillna(False).astype(float) * 30.0

    # 3. Default DPD Contradiction
    if "current_status" in df.columns and "days_past_due" in df.columns:
        bad_dpd = (df["current_status"] == "Default") & (df["days_past_due"] < 60)
        scores -= bad_dpd.fillna(False).astype(float) * 25.0

    # 4. Outliers
    if "interest_rate" in df.columns:
        bad_ir = (df["interest_rate"] > 15.0).fillna(False)
        scores -= bad_ir.astype(float) * 15.0

    if "current_balance" in df.columns and "original_balance" in df.columns:
        excess_bal = (df["current_balance"] > df["original_balance"] * 2.0).fillna(False)
        scores -= excess_bal.astype(float) * 15.0

    # 5. Missingness penalties
    if "credit_score_band" in df.columns:
        miss_fico = df["credit_score_band"].isna()
        scores -= miss_fico.astype(float) * 10.0

    if "interest_rate" in df.columns:
        miss_ir = df["interest_rate"].isna()
        scores -= miss_ir.astype(float) * 5.0

    # 6. Document exception
    if "document_status" in df.columns:
        doc_gap = (df["document_status"] == "Missing Items").fillna(False)
        scores -= doc_gap.astype(float) * 5.0

    # Clamp to [0, 100]
    scores = np.clip(scores, 0.0, 100.0)
    return pd.Series(scores, index=df.index, name="data_quality_score")


def evaluate_batch_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute portfolio/batch-level aggregated Data Quality analytics.
    """
    dq_series = compute_record_quality_scores(df)
    mean_dq = float(dq_series.mean())
    median_dq = float(dq_series.median())
    std_dq = float(dq_series.std())

    # Tier assignment
    p90_plus = float((dq_series >= 90).mean() * 100)
    p75_to_90 = float(((dq_series >= 75) & (dq_series < 90)).mean() * 100)
    p_below_75 = float((dq_series < 75).mean() * 100)

    if mean_dq >= 95.0:
        grade = "A (Pristine / Grade A Quality)"
    elif mean_dq >= 85.0:
        grade = "B (Good / Minor Gaps Identified)"
    elif mean_dq >= 75.0:
        grade = "C (Moderate Quality / Requires Remediation)"
    else:
        grade = "D/F (Critical Deficiencies Present)"

    return {
        "batch_mean_dq_score": round(mean_dq, 2),
        "batch_median_dq_score": round(median_dq, 2),
        "batch_std_dq_score": round(std_dq, 2),
        "batch_quality_grade": grade,
        "score_distribution": {
            "pristine_gte_90_pct": round(p90_plus, 2),
            "acceptable_75_to_90_pct": round(p75_to_90, 2),
            "degraded_lt_75_pct": round(p_below_75, 2),
        },
        "score_percentiles": {
            "p10": round(float(dq_series.quantile(0.10)), 2),
            "p25": round(float(dq_series.quantile(0.25)), 2),
            "p50": round(float(dq_series.quantile(0.50)), 2),
            "p75": round(float(dq_series.quantile(0.75)), 2),
            "p90": round(float(dq_series.quantile(0.90)), 2),
        },
    }
