"""
Cross-Column Relationship Break Detection Module
================================================
Audits logical consistency across multiple interdependent tabular features
(status vs. balance vs. delinquency flags vs. loss severity).
"""

from typing import Dict, Any, List
import pandas as pd


def detect_relationship_breaks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate deterministic domain invariants to identify semantic contradictions in loan panel data.
    """
    total_rows = len(df)
    results: Dict[str, Any] = {
        "total_rows_inspected": total_rows,
        "total_violations": 0,
        "violation_types": {},
    }

    checks = []

    # Check 1: Paid Off with positive non-zero balance (> $1,000)
    if "current_status" in df.columns and "current_balance" in df.columns:
        mask = (df["current_status"] == "Paid Off") & (df["current_balance"] > 1000)
        checks.append({
            "code": "BRK01_PAID_OFF_BALANCE",
            "name": "Paid Off Status with Active Non-Zero Balance",
            "condition": "current_status == 'Paid Off' AND current_balance > 1000",
            "severity": "CRITICAL",
            "mask": mask,
            "description": "Loan is flagged as Paid Off but retains a substantial outstanding principal balance.",
        })

    # Check 2: Prepaid with positive balance (> $1,000)
    if "current_status" in df.columns and "current_balance" in df.columns:
        mask = (df["current_status"] == "Prepaid") & (df["current_balance"] > 1000)
        checks.append({
            "code": "BRK02_PREPAID_BALANCE",
            "name": "Prepaid Status with Active Non-Zero Balance",
            "condition": "current_status == 'Prepaid' AND current_balance > 1000",
            "severity": "HIGH",
            "mask": mask,
            "description": "Loan is flagged as Prepaid but retains positive current balance.",
        })

    # Check 3: Default status but days_past_due < 60
    if "current_status" in df.columns and "days_past_due" in df.columns:
        mask = (df["current_status"] == "Default") & (df["days_past_due"] < 60)
        checks.append({
            "code": "BRK03_DEFAULT_LOW_DPD",
            "name": "Default Status with Low Days Past Due",
            "condition": "current_status == 'Default' AND days_past_due < 60",
            "severity": "HIGH",
            "mask": mask,
            "description": "Loan is classified as Default without reaching the standard 60-90 DPD threshold.",
        })

    # Check 4: Current status with active delinquency DPD (> 30)
    if "current_status" in df.columns and "days_past_due" in df.columns:
        mask = (df["current_status"] == "Current") & (df["days_past_due"] >= 30)
        checks.append({
            "code": "BRK04_CURRENT_HIGH_DPD",
            "name": "Current Status with Active Delinquency",
            "condition": "current_status == 'Current' AND days_past_due >= 30",
            "severity": "HIGH",
            "mask": mask,
            "description": "Loan is classified as Current despite having >= 30 days past due.",
        })

    # Check 5: Default status with default_flag == 0
    if "current_status" in df.columns and "default_flag" in df.columns:
        mask = (df["current_status"] == "Default") & (df["default_flag"] == 0)
        checks.append({
            "code": "BRK05_STATUS_FLAG_MISMATCH",
            "name": "Default Status without Default Flag",
            "condition": "current_status == 'Default' AND default_flag == 0",
            "severity": "CRITICAL",
            "mask": mask,
            "description": "Discrepancy between categorical status and binary default indicator.",
        })

    # Check 6: Modified loan with Missing Documentation
    if "modification_flag" in df.columns and "document_status" in df.columns:
        mask = (df["modification_flag"] == 1) & (df["document_status"] == "Missing Items")
        checks.append({
            "code": "BRK06_MOD_MISSING_DOC",
            "name": "Restructured/Modified Loan with Missing Verification Files",
            "condition": "modification_flag == 1 AND document_status == 'Missing Items'",
            "severity": "MEDIUM",
            "mask": mask,
            "description": "Restructured loan lacks required trailing modification documentation.",
        })

    # Check 7: Current balance > 2x original balance
    if "current_balance" in df.columns and "original_balance" in df.columns:
        mask = (df["current_balance"] > df["original_balance"] * 2.0)
        checks.append({
            "code": "BRK07_EXCESSIVE_BALANCE_GROWTH",
            "name": "Current Balance Exceeds 200% Original Disbursement",
            "condition": "current_balance > 2.0 * original_balance",
            "severity": "HIGH",
            "mask": mask,
            "description": "Unpaid principal balance exceeds twice the original note balance without recorded negative amortization terms.",
        })

    all_violation_indices = set()

    for item in checks:
        mask = item["mask"].fillna(False)
        v_count = int(mask.sum())
        all_violation_indices.update(df[mask].index)

        examples = []
        if v_count > 0:
            sample_cols = [c for c in ["loan_id", "reporting_month", "current_status", "current_balance", "days_past_due", "modification_flag", "document_status"] if c in df.columns]
            examples = df.loc[mask, sample_cols].head(3).to_dict(orient="records")

        results["violation_types"][item["code"]] = {
            "name": item["name"],
            "condition": item["condition"],
            "severity": item["severity"],
            "count": v_count,
            "pct": round(v_count / total_rows * 100, 4),
            "description": item["description"],
            "examples": examples,
        }

    results["total_violations"] = len(all_violation_indices)
    results["total_violations_pct"] = round(len(all_violation_indices) / total_rows * 100, 4)

    return results
