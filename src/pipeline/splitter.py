"""
Time-Aware Dataset Splitter Module
==================================
Performs strict time-aware cohort splitting with automated verification
against loan_id data leakage across train, validation, and test partitions.
"""

from typing import Tuple, Dict, Any, Optional
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def time_aware_cohort_split(
    df: pd.DataFrame,
    val_cutoff: str = "2020-01-01",
    test_cutoff: str = "2022-01-01",
    time_col: str = "origination_month",
    id_col: str = "loan_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split loan panel dataset into Train, Validation, and Test cohorts based on origination time.
    
    Temporal partition:
      - Train: origination_month < val_cutoff (e.g. < 2020-01)
      - Validation: val_cutoff <= origination_month < test_cutoff (e.g. 2020-01 to 2021-12)
      - Test: origination_month >= test_cutoff (e.g. >= 2022-01)
    
    Strict Invariant:
      Intersection(Train_IDs, Val_IDs) == Ø
      Intersection(Train_IDs, Test_IDs) == Ø
      Intersection(Val_IDs, Test_IDs) == Ø
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe.")
    if id_col not in df.columns:
        raise ValueError(f"Identifier column '{id_col}' not found in dataframe.")

    dates = pd.to_datetime(df[time_col].astype(str) + "-01", errors="coerce")
    val_dt = pd.Timestamp(val_cutoff)
    test_dt = pd.Timestamp(test_cutoff)

    train_mask = dates < val_dt
    val_mask = (dates >= val_dt) & (dates < test_dt)
    test_mask = dates >= test_dt

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Integrity verification
    train_ids = set(train_df[id_col].unique())
    val_ids = set(val_df[id_col].unique())
    test_ids = set(test_df[id_col].unique())

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED in time-aware split: "
            f"Train-Val overlap={len(overlap_train_val)}, "
            f"Train-Test overlap={len(overlap_train_test)}, "
            f"Val-Test overlap={len(overlap_val_test)}"
        )

    log.info(
        f"Time-aware split complete: "
        f"Train={len(train_df):,} rows ({len(train_ids):,} loans), "
        f"Val={len(val_df):,} rows ({len(val_ids):,} loans), "
        f"Test={len(test_df):,} rows ({len(test_ids):,} loans)"
    )

    return train_df, val_df, test_df


def audit_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    id_col: str = "loan_id",
    time_col: str = "origination_month",
) -> Dict[str, Any]:
    """
    Audit and return report of zero-leakage assertions.
    """
    train_ids = set(train_df[id_col].unique())
    val_ids = set(val_df[id_col].unique())
    test_ids = set(test_df[id_col].unique()) if test_df is not None else set()

    tv_overlap = len(train_ids & val_ids)
    tt_overlap = len(train_ids & test_ids)
    vt_overlap = len(val_ids & test_ids)

    # Check temporal ordering
    t_max_date = pd.to_datetime(train_df[time_col] + "-01").max()
    v_min_date = pd.to_datetime(val_df[time_col] + "-01").min()

    temporal_valid = t_max_date < v_min_date

    is_leakage_free = (tv_overlap == 0) and (tt_overlap == 0) and (vt_overlap == 0) and temporal_valid

    return {
        "is_leakage_free": is_leakage_free,
        "train_val_overlap_count": tv_overlap,
        "train_test_overlap_count": tt_overlap,
        "val_test_overlap_count": vt_overlap,
        "train_max_origination": str(t_max_date.strftime("%Y-%m")),
        "val_min_origination": str(v_min_date.strftime("%Y-%m")),
        "temporal_order_valid": temporal_valid,
    }
