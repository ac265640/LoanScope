"""
Anomaly Drivers & Reviewer Case Generator
=========================================
Extracts top contributing driver features for anomalous records and compiles
at least 20 detailed, reviewer-ready anomaly case files with plain-English explanations.
"""

import sys
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.models.anomaly.isolation_forest import predict_anomaly_scores
from src.models.anomaly.exception_predictor import compute_rule_violation_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def identify_top_anomaly_drivers(row: pd.Series, baseline_medians: pd.Series, feature_cols: List[str]) -> List[str]:
    """Identify features with the largest relative standardized deviation from portfolio baseline."""
    deviations = {}
    for col in feature_cols:
        val = row.get(col)
        base = baseline_medians.get(col, 0)
        if pd.api.types.is_numeric_dtype(type(val)) and base is not None:
            dev = abs(float(val) - float(base)) / (abs(float(base)) + 1e-4)
            deviations[col] = dev

    top_drivers = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:3]
    return [d[0] for d in top_drivers]


def generate_reviewer_anomaly_cases(df: pd.DataFrame, n_cases: int = 25) -> List[Dict[str, Any]]:
    """Extract and format >= 20 detailed reviewer-ready anomaly examples."""
    log.info(f"Generating {n_cases} reviewer-ready anomaly case records...")

    feat_cols = get_feature_columns()
    medians = df[feat_cols].median()

    # Sort by anomaly_score descending and filter to anomalous rows
    anom_sorted = df.sort_values("anomaly_score", ascending=False)
    candidates = anom_sorted.head(n_cases * 2)

    cases = []
    for _, row in candidates.iterrows():
        drivers = identify_top_anomaly_drivers(row, medians, feat_cols)

        # Build natural-language plain-English explanation
        status = row.get("current_status", "Unknown")
        bal = row.get("current_balance", 0.0)
        dpd = row.get("days_past_due", 0)
        ir = row.get("interest_rate_imputed", 0.0)
        etype = row.get("exception_type", "General Outlier")

        reasons = []
        if status == "Paid Off" and bal > 1000:
            reasons.append(f"Loan is marked Paid Off but retains outstanding principal balance of ${bal:,.2f}.")
        if status == "Default" and dpd < 60:
            reasons.append(f"Classified as Default despite low recorded delinquency of {dpd} DPD.")
        if ir > 12.0:
            reasons.append(f"Unusually elevated note rate of {ir:.2f}% (portfolio median: {medians.get('interest_rate_imputed', 4.5):.2f}%).")
        if dpd > 180:
            reasons.append(f"Severe chronic delinquency with {dpd} days past due.")
        if row.get("balance_to_orig_ratio", 1.0) > 1.8:
            reasons.append(f"Outstanding balance is {row.get('balance_to_orig_ratio'):.1f}x original disbursement.")

        if not reasons:
            reasons.append(f"Multivariate deviation across primary drivers: {', '.join(drivers)}.")

        explanation = " ".join(reasons)

        recommended_action = "Manual Servicer Reconciliation & Data Audit"
        if "Paid Off" in explanation:
            recommended_action = "Reconcile final payoff wire with servicer cash ledger"
        elif "Default" in explanation:
            recommended_action = "Audit foreclosure/charge-off legal timestamp"
        elif "note rate" in explanation:
            recommended_action = "Verify note rate against loan agreement schedule"

        cases.append({
            "loan_id": row["loan_id"],
            "reporting_month": row["reporting_month"],
            "current_status": status,
            "current_balance": round(float(bal), 2),
            "interest_rate": round(float(ir), 2),
            "days_past_due": int(dpd),
            "exception_type": etype,
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "top_drivers": drivers,
            "plain_english_explanation": explanation,
            "recommended_action": recommended_action,
        })

        if len(cases) >= n_cases:
            break

    return cases


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)
    feat_df["anomaly_score"] = predict_anomaly_scores(feat_df)

    cases = generate_reviewer_anomaly_cases(feat_df, n_cases=25)

    # Save to JSON
    out_json = MODELS_DIR / "reviewer_anomaly_cases.json"
    with open(out_json, "w") as f:
        json.dump(cases, f, indent=2)

    # Write Markdown section to reports
    out_md = REPORTS_DIR / "anomaly_reviewer_cases.md"
    with open(out_md, "w") as f:
        f.write("# Anomaly & Exception Reviewer Cases\n\n")
        f.write(f"**Total Reviewer Examples Documented**: `{len(cases)}`\n\n")
        f.write("| # | Loan ID | Reporting Month | Status | Balance ($) | DPD | Anomaly Score | Exception Type | Top Driver Features | Recommended Action |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for i, c in enumerate(cases, 1):
            f.write(f"| {i} | `{c['loan_id']}` | `{c['reporting_month']}` | `{c['current_status']}` | `${c['current_balance']:,.2f}` | `{c['days_past_due']}` | `{c['anomaly_score']:.4f}` | **`{c['exception_type']}`** | `{', '.join(c['top_drivers'])}` | {c['recommended_action']} |\n")

        f.write("\n\n## Detailed Case Rationale Breakdown\n\n")
        for i, c in enumerate(cases, 1):
            f.write(f"### Case {i}: Loan `{c['loan_id']}` (Score: `{c['anomaly_score']:.4f}`)\n")
            f.write(f"- **Exception Category**: `{c['exception_type']}`\n")
            f.write(f"- **Current Status**: `{c['current_status']}` | **Balance**: `${c['current_balance']:,.2f}` | **DPD**: `{c['days_past_due']}`\n")
            f.write(f"- **Diagnostic Explanation**: {c['plain_english_explanation']}\n")
            f.write(f"- **Top Contributing Features**: `{', '.join(c['top_drivers'])}`\n")
            f.write(f"- **Reviewer Action**: **{c['recommended_action']}**\n\n")

    log.info(f"✅ Generated {len(cases)} anomaly cases -> saved to {out_md}")


if __name__ == "__main__":
    main()
