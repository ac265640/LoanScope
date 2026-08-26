"""
Defensive Input Validation & Schema Guardrails
==============================================
Provides robust, descriptive input data validation across all pipeline entrypoints.
Intercepts schema discrepancies, missing required columns, corrupted types,
or empty datasets before model ingestion with user-friendly diagnostics.

Usage:
  from src.pipeline.validator import validate_raw_dataset, validate_features_dataframe, DataValidationError
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANDATORY_TRAIN_COLUMNS = [
    "loan_id", "month_index", "reporting_month", "origination_month",
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "credit_score_band",
    "current_status", "days_past_due", "modification_flag",
]

MANDATORY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
]


class DataValidationError(Exception):
    """Raised when an input dataset fails schema or data invariant validation."""
    pass


def validate_raw_dataset(
    df: pd.DataFrame,
    dataset_name: str = "train",
    require_targets: bool = True,
) -> bool:
    """
    Validate raw tabular loan panel DataFrame before processing.
    Raises DataValidationError with actionable diagnostics on failure.
    """
    if df is None or len(df) == 0:
        raise DataValidationError(f"[{dataset_name}] Dataset is empty or None.")

    # 1. Missing columns check
    missing_cols = [c for c in MANDATORY_TRAIN_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"[{dataset_name}] Missing required columns: {missing_cols}. "
            f"Please verify schema matches data/data_dictionary.md"
        )

    if require_targets:
        missing_targets = [t for t in MANDATORY_TARGETS if t not in df.columns]
        if missing_targets:
            raise DataValidationError(
                f"[{dataset_name}] Missing required prediction target columns: {missing_targets}."
            )

    # 2. Date format checks
    for date_col in ["reporting_month", "origination_month"]:
        if date_col in df.columns:
            sample_dates = df[date_col].dropna().head(100)
            invalid_dates = sample_dates[~sample_dates.astype(str).str.match(r"^\d{4}-\d{2}$")]
            if len(invalid_dates) > 0:
                raise DataValidationError(
                    f"[{dataset_name}] Column '{date_col}' contains invalid date formats (expected YYYY-MM): {invalid_dates.iloc[0]}"
                )

    # 3. Numeric bounds checks
    if "original_balance" in df.columns:
        neg_bal = (df["original_balance"] < 0).sum()
        if neg_bal > 0:
            raise DataValidationError(f"[{dataset_name}] Found {neg_bal} records with negative original_balance.")

    if "days_past_due" in df.columns:
        neg_dpd = (df["days_past_due"] < 0).sum()
        if neg_dpd > 0:
            raise DataValidationError(f"[{dataset_name}] Found {neg_dpd} records with negative days_past_due.")

    log.info(f"✅ [{dataset_name}] Passed defensive schema validation ({len(df):,} records, {len(df.columns)} columns).")
    return True


def validate_features_dataframe(
    df: pd.DataFrame,
    required_features: List[str],
    dataset_name: str = "features",
) -> bool:
    """Validate engineered feature matrix before model training or scoring."""
    if df is None or len(df) == 0:
        raise DataValidationError(f"[{dataset_name}] Feature DataFrame is empty.")

    missing_feats = [f for f in required_features if f not in df.columns]
    if missing_feats:
        raise DataValidationError(
            f"[{dataset_name}] Missing engineered feature columns: {missing_feats}."
        )

    # Check for complete NaN columns
    nan_cols = [f for f in required_features if df[f].isna().all()]
    if nan_cols:
        raise DataValidationError(
            f"[{dataset_name}] Feature columns contain 100% NaN values: {nan_cols}."
        )

    log.info(f"✅ [{dataset_name}] Feature matrix validated successfully ({len(required_features)} features, {len(df):,} rows).")
    return True
