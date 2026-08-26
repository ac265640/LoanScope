"""
Unit Tests for Defensive Input Validator
=========================================
Verifies that `src/pipeline/validator.py` correctly validates compliant datasets
and raises informative `DataValidationError` exceptions for malformed inputs.
"""

import pytest
import pandas as pd
from src.pipeline.validator import validate_raw_dataset, validate_features_dataframe, DataValidationError


def test_validator_passes_on_valid_data():
    valid_df = pd.DataFrame([{
        "loan_id": "LN0000001",
        "month_index": 1,
        "reporting_month": "2020-01",
        "origination_month": "2019-01",
        "loan_age_months": 12,
        "remaining_term_months": 348,
        "original_balance": 300000.0,
        "current_balance": 295000.0,
        "interest_rate": 4.5,
        "credit_score_band": "700-739",
        "current_status": "Current",
        "days_past_due": 0,
        "modification_flag": 0,
        "next_3m_delinquency_flag": 0,
        "next_6m_delinquency_flag": 0,
        "next_12m_default_flag": 0,
        "next_12m_prepayment_flag": 0,
        "next_state": "Current",
    }])
    assert validate_raw_dataset(valid_df, "test_valid", require_targets=True) is True


def test_validator_fails_on_missing_column():
    invalid_df = pd.DataFrame([{
        "loan_id": "LN0000001",
        "month_index": 1,
        "reporting_month": "2020-01",
        # missing original_balance, etc.
    }])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw_dataset(invalid_df, "test_invalid")


def test_validator_fails_on_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="is empty"):
        validate_raw_dataset(empty_df, "test_empty")


def test_feature_validator_fails_on_all_nan():
    nan_feat_df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "feat_b": [None, None, None],
    })
    with pytest.raises(DataValidationError, match="100% NaN"):
        validate_features_dataframe(nan_feat_df, ["feat_a", "feat_b"])
