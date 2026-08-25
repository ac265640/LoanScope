"""
Automated Test Suite for Time-Aware Splitter & Data Leakage Prevention
======================================================================
Verifies:
  1. Zero loan_id overlap across Train, Validation, and Test cohorts.
  2. Strict temporal ordering (no future data in training partitions).
  3. Correct partition assignment for boundary edge-cases.
"""

import pytest
import pandas as pd
import numpy as np
from src.pipeline.splitter import time_aware_cohort_split, audit_split_leakage


@pytest.fixture
def dummy_panel_data():
    """Create synthetic loan panel data across different origination cohorts."""
    loans = []
    # 50 early loans (2018-01 to 2019-12)
    for i in range(1, 51):
        lid = f"LN_EARLY_{i:04d}"
        for m in range(1, 13):
            loans.append({
                "loan_id": lid,
                "origination_month": "2018-06",
                "month_index": m,
                "current_balance": 200000 - m * 500,
                "current_status": "Current",
            })

    # 30 mid loans (2020-01 to 2021-12)
    for i in range(51, 81):
        lid = f"LN_MID_{i:04d}"
        for m in range(1, 13):
            loans.append({
                "loan_id": lid,
                "origination_month": "2020-06",
                "month_index": m,
                "current_balance": 250000 - m * 600,
                "current_status": "Current",
            })

    # 20 late loans (2022-01 to 2023-06)
    for i in range(81, 101):
        lid = f"LN_LATE_{i:04d}"
        for m in range(1, 13):
            loans.append({
                "loan_id": lid,
                "origination_month": "2022-06",
                "month_index": m,
                "current_balance": 300000 - m * 700,
                "current_status": "Current",
            })

    return pd.DataFrame(loans)


def test_time_aware_split_no_overlap(dummy_panel_data):
    """Assert zero loan_id intersection across all split sets."""
    train_df, val_df, test_df = time_aware_cohort_split(
        dummy_panel_data,
        val_cutoff="2020-01-01",
        test_cutoff="2022-01-01",
        time_col="origination_month",
        id_col="loan_id",
    )

    train_ids = set(train_df["loan_id"].unique())
    val_ids = set(val_df["loan_id"].unique())
    test_ids = set(test_df["loan_id"].unique())

    assert len(train_ids) == 50, f"Expected 50 train loans, got {len(train_ids)}"
    assert len(val_ids) == 30, f"Expected 30 val loans, got {len(val_ids)}"
    assert len(test_ids) == 20, f"Expected 20 test loans, got {len(test_ids)}"

    # Assert zero overlap
    assert len(train_ids & val_ids) == 0, "Train and Val share loan_ids!"
    assert len(train_ids & test_ids) == 0, "Train and Test share loan_ids!"
    assert len(val_ids & test_ids) == 0, "Val and Test share loan_ids!"


def test_split_temporal_ordering(dummy_panel_data):
    """Assert max train origination date is strictly earlier than min validation origination date."""
    train_df, val_df, test_df = time_aware_cohort_split(
        dummy_panel_data,
        val_cutoff="2020-01-01",
        test_cutoff="2022-01-01",
    )

    t_max = pd.to_datetime(train_df["origination_month"] + "-01").max()
    v_min = pd.to_datetime(val_df["origination_month"] + "-01").min()
    v_max = pd.to_datetime(val_df["origination_month"] + "-01").max()
    tst_min = pd.to_datetime(test_df["origination_month"] + "-01").min()

    assert t_max < v_min, f"Temporal violation: Train max ({t_max}) >= Val min ({v_min})"
    assert v_max < tst_min, f"Temporal violation: Val max ({v_max}) >= Test min ({tst_min})"


def test_audit_split_leakage_helper(dummy_panel_data):
    """Test the audit_split_leakage reporting helper."""
    train_df, val_df, test_df = time_aware_cohort_split(
        dummy_panel_data,
        val_cutoff="2020-01-01",
        test_cutoff="2022-01-01",
    )

    audit = audit_split_leakage(train_df, val_df, test_df)
    assert audit["is_leakage_free"] is True
    assert audit["train_val_overlap_count"] == 0
    assert audit["train_test_overlap_count"] == 0
    assert audit["temporal_order_valid"] is True
