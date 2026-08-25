"""
Source Conflict Detection and Reconciliation Module
===================================================
Reconciles primary loan performance records against secondary servicer update feeds
to detect feed-level discrepancies, stale reporting, and conflicting balances/statuses.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def detect_source_conflicts(
    main_df: pd.DataFrame,
    servicer_df: pd.DataFrame,
    balance_tolerance: float = 100.0,
) -> Dict[str, Any]:
    """
    Perform deep join and discrepancy audit between primary panel and secondary servicer feed.
    """
    join_keys = ["loan_id", "reporting_month"]
    merged = pd.merge(
        main_df,
        servicer_df,
        on=join_keys,
        suffixes=("_primary", "_servicer"),
        how="inner",
    )

    n_matched = len(merged)
    if n_matched == 0:
        return {
            "n_servicer_records": len(servicer_df),
            "n_matched_records": 0,
            "match_rate_pct": 0.0,
            "status": "No overlapping records found.",
        }

    # 1. Balance conflicts
    bal_diff = (merged["current_balance_primary"] - merged["current_balance_servicer"]).abs()
    bal_conflict_mask = bal_diff > balance_tolerance
    n_bal_conflicts = int(bal_conflict_mask.sum())

    # 2. Status conflicts
    status_conflict_mask = merged["current_status_primary"] != merged["current_status_servicer"]
    n_status_conflicts = int(status_conflict_mask.sum())

    # 3. DPD conflicts (if present)
    dpd_conflicts = 0
    if "days_past_due_servicer" in merged.columns and "days_past_due_primary" in merged.columns:
        dpd_diff = (merged["days_past_due_primary"] - merged["days_past_due_servicer"]).abs()
        dpd_conflicts = int((dpd_diff > 0).sum())

    # Total conflicting rows
    any_conflict = bal_conflict_mask | status_conflict_mask
    n_total_conflicts = int(any_conflict.sum())

    # Extract sample conflicts for reviewer evidence
    sample_conflicts = []
    if n_total_conflicts > 0:
        conflict_rows = merged[any_conflict].head(10)
        for _, row in conflict_rows.iterrows():
            sample_conflicts.append({
                "loan_id": row["loan_id"],
                "reporting_month": row["reporting_month"],
                "primary_balance": float(row["current_balance_primary"]),
                "servicer_balance": float(row["current_balance_servicer"]),
                "balance_diff": round(float(row["current_balance_primary"] - row["current_balance_servicer"]), 2),
                "primary_status": str(row["current_status_primary"]),
                "servicer_status": str(row["current_status_servicer"]),
            })

    return {
        "n_servicer_records": len(servicer_df),
        "n_matched_records": n_matched,
        "match_rate_pct": round(n_matched / len(servicer_df) * 100, 2),
        "total_conflicting_records": n_total_conflicts,
        "conflict_rate_pct": round(n_total_conflicts / n_matched * 100, 2),
        "balance_discrepancies_count": n_bal_conflicts,
        "balance_discrepancies_pct": round(n_bal_conflicts / n_matched * 100, 2),
        "status_discrepancies_count": n_status_conflicts,
        "status_discrepancies_pct": round(n_status_conflicts / n_matched * 100, 2),
        "dpd_discrepancies_count": dpd_conflicts,
        "sample_conflicts": sample_conflicts,
    }
