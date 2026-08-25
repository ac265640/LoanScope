"""
Unit Tests for Feature Engineering Pipeline
============================================
Verifies non-leakage properties, rolling calculation consistency,
ordinal mapping stability, and handling of missing fields in feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import engineer_panel_features, get_feature_columns


@pytest.fixture
def mini_panel_df():
    """Create a synthetic 2-loan 6-month panel dataset for deterministic feature testing."""
    records = []
    # Loan 1: Clean amortizing current loan
    for m in range(1, 7):
        records.append({
            "loan_id": "LN0000001",
            "month_index": m,
            "reporting_month": f"2020-{m:02d}",
            "origination_month": "2020-01",
            "loan_age_months": m,
            "remaining_term_months": 360 - m,
            "original_balance": 300000.0,
            "current_balance": 300000.0 - (m * 500.0),
            "interest_rate": 4.5,
            "credit_score_band": "700-739",
            "ltv_band": "70-80%",
            "dti_band": "28-36%",
            "state": "CA",
            "loan_purpose": "Purchase",
            "occupancy_type": "Primary",
            "property_type": "Single Family",
            "servicer_name": "Servicer Alpha",
            "current_status": "Current",
            "days_past_due": 0,
            "modification_flag": 0,
            "prepayment_flag": 0,
            "default_flag": 0,
            "loss_severity_band": "None",
            "last_updated_at": f"2020-{m:02d}-28",
            "source_system": "Core_Servicing",
            "document_status": "Complete",
        })

    # Loan 2: Deteriorating delinquent loan
    for m in range(1, 7):
        dpd = 0 if m <= 2 else (m - 2) * 30
        status = "Current" if dpd == 0 else f"{dpd-29}-{dpd} DPD"
        records.append({
            "loan_id": "LN0000002",
            "month_index": m,
            "reporting_month": f"2020-{m:02d}",
            "origination_month": "2020-01",
            "loan_age_months": m,
            "remaining_term_months": 360 - m,
            "original_balance": 200000.0,
            "current_balance": 200000.0,
            "interest_rate": 6.5,
            "credit_score_band": "<620",
            "ltv_band": "90-100%",
            "dti_band": ">43%",
            "state": "FL",
            "loan_purpose": "Refinance",
            "occupancy_type": "Primary",
            "property_type": "Condo",
            "servicer_name": "Servicer Beta",
            "current_status": status,
            "days_past_due": dpd,
            "modification_flag": 1 if m >= 4 else 0,
            "prepayment_flag": 0,
            "default_flag": 1 if dpd >= 90 else 0,
            "loss_severity_band": "None",
            "last_updated_at": f"2020-{m:02d}-28",
            "source_system": "Secondary_Feed",
            "document_status": "Missing Items",
        })

    return pd.DataFrame(records)


def test_feature_columns_list():
    """Verify get_feature_columns returns a non-empty list of unique column strings."""
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) >= 25
    assert len(cols) == len(set(cols)), "Feature column list contains duplicates"


def test_engineer_features_preserves_row_count(mini_panel_df):
    """Engineered dataset must retain exact same row count."""
    feat_df = engineer_panel_features(mini_panel_df)
    assert len(feat_df) == len(mini_panel_df)


def test_no_future_information_in_rolling_dpd(mini_panel_df):
    """Month 1 rolling delinquent count must strictly reflect only past information."""
    feat_df = engineer_panel_features(mini_panel_df)
    m1_loan2 = feat_df[(feat_df["loan_id"] == "LN0000002") & (feat_df["month_index"] == 1)].iloc[0]
    # In month 1, borrower was Current, so rolling max DPD should be 0
    assert m1_loan2["dpd_roll_max_3m"] == 0.0
    assert m1_loan2["dpd_roll_max_6m"] == 0.0


def test_balance_to_orig_ratio_calculation(mini_panel_df):
    """balance_to_orig_ratio must correctly compute current_balance / original_balance."""
    feat_df = engineer_panel_features(mini_panel_df)
    m6_loan1 = feat_df[(feat_df["loan_id"] == "LN0000001") & (feat_df["month_index"] == 6)].iloc[0]
    expected_ratio = (300000.0 - 3000.0) / 300000.0
    assert np.isclose(m6_loan1["balance_to_orig_ratio"], expected_ratio, atol=1e-4)


def test_ordinal_encodings_valid(mini_panel_df):
    """Categorical ordinal columns must produce monotonic integer-like orderings."""
    feat_df = engineer_panel_features(mini_panel_df)
    loan1 = feat_df[feat_df["loan_id"] == "LN0000001"].iloc[0]
    loan2 = feat_df[feat_df["loan_id"] == "LN0000002"].iloc[0]

    # Prime credit (720-739) should have higher ordinal score than Subprime (<620)
    assert loan1["credit_score_ordinal"] > loan2["credit_score_ordinal"]
    # Low DTI (<35%) should have lower ordinal risk than High DTI (>43%)
    assert loan1["dti_ordinal"] < loan2["dti_ordinal"]


def test_no_all_nan_feature_columns(mini_panel_df):
    """Engineered feature columns should not contain entirely NaN series."""
    feat_df = engineer_panel_features(mini_panel_df)
    features = get_feature_columns()
    for col in features:
        assert col in feat_df.columns, f"Feature {col} missing from output DataFrame"
        assert not feat_df[col].isna().all(), f"Feature {col} is 100% NaN"
