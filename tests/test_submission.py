"""
Automated Submission Format & Value Range Validator
====================================================
Asserts that `submission/submission.csv` strictly conforms to the challenge requirements:
- Exact column-for-column alignment with `submission_template.csv`
- Complete loan ID parity with the test cohort (zero missing or extraneous loans)
- Numerical probability and anomaly score bounds [0.0, 1.0]
- Zero NaNs in mandatory prediction fields
- Valid categorical values for next_state and exception_required
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_CSV = SUBMISSION_DIR / "submission.csv"
TEMPLATE_CSV = SUBMISSION_DIR / "submission_template.csv"

VALID_STATES = [
    "Current", "30-59 DPD", "60-89 DPD", "90+ DPD",
    "Default", "Prepaid", "Paid Off",
]

PROB_COLUMNS = [
    "prob_next_3m_delinquency",
    "prob_next_6m_delinquency",
    "prob_next_12m_default",
    "prob_next_12m_prepayment",
]


@pytest.fixture(scope="module")
def submission_df():
    if not SUBMISSION_CSV.exists():
        pytest.fail(f"submission.csv does not exist at {SUBMISSION_CSV}. Run `make submission` first.")
    df = pd.read_csv(SUBMISSION_CSV)
    assert len(df) > 0, "submission.csv is empty."
    return df


@pytest.fixture(scope="module")
def template_df():
    if not TEMPLATE_CSV.exists():
        pytest.skip("submission_template.csv not found.")
    return pd.read_csv(TEMPLATE_CSV)


def test_submission_file_exists():
    """submission.csv must exist and have non-zero file size."""
    assert SUBMISSION_CSV.exists(), f"Missing submission.csv at {SUBMISSION_CSV}"
    assert SUBMISSION_CSV.stat().st_size > 1000, "submission.csv file size is suspiciously small"


def test_submission_column_parity_with_template(submission_df, template_df):
    """submission.csv must have the exact same columns in the exact same order as the template."""
    sub_cols = list(submission_df.columns)
    tmpl_cols = list(template_df.columns)
    assert sub_cols == tmpl_cols, f"Column mismatch!\nSubmission: {sub_cols}\nTemplate:   {tmpl_cols}"


def test_submission_row_count_parity(submission_df, template_df):
    """submission.csv must have the exact same number of rows as submission_template.csv."""
    assert len(submission_df) == len(template_df), (
        f"Row count mismatch: submission has {len(submission_df)} rows, template has {len(template_df)}"
    )


def test_submission_loan_id_parity(submission_df, template_df):
    """submission.csv loan_id set must exactly match the template loan_id set."""
    sub_ids = set(submission_df["loan_id"])
    tmpl_ids = set(template_df["loan_id"])
    assert sub_ids == tmpl_ids, (
        f"Loan ID mismatch! Extra: {len(sub_ids - tmpl_ids)}, Missing: {len(tmpl_ids - sub_ids)}"
    )


def test_no_null_loan_identifiers(submission_df):
    """No nulls allowed in loan_id or reporting_month."""
    assert submission_df["loan_id"].isna().sum() == 0, "Found NaNs in loan_id"
    assert submission_df["reporting_month"].isna().sum() == 0, "Found NaNs in reporting_month"


def test_probabilities_in_valid_range(submission_df):
    """All predicted probability columns must be floats in [0.0, 1.0] with zero NaNs."""
    for col in PROB_COLUMNS:
        assert col in submission_df.columns, f"Missing probability column {col}"
        vals = submission_df[col]
        assert vals.isna().sum() == 0, f"Found NaNs in {col}"
        assert (vals >= 0.0).all(), f"Found negative values in {col}"
        assert (vals <= 1.0).all(), f"Found values > 1.0 in {col}"
        # Assert non-degenerate predictions (standard deviation > 0)
        assert vals.std() > 0.001, f"{col} has degenerate constant predictions"


def test_anomaly_score_in_valid_range(submission_df):
    """anomaly_score must be in [0.0, 1.0] with zero NaNs."""
    assert "anomaly_score" in submission_df.columns
    scores = submission_df["anomaly_score"]
    assert scores.isna().sum() == 0, "Found NaNs in anomaly_score"
    assert (scores >= 0.0).all(), "Found negative anomaly_score"
    assert (scores <= 1.0).all(), "Found anomaly_score > 1.0"


def test_confidence_in_valid_range(submission_df):
    """confidence score must be in [0.0, 1.0] with zero NaNs."""
    assert "confidence" in submission_df.columns
    conf = submission_df["confidence"]
    assert conf.isna().sum() == 0, "Found NaNs in confidence"
    assert (conf >= 0.0).all(), "Found negative confidence"
    assert (conf <= 1.0).all(), "Found confidence > 1.0"


def test_exception_required_binary(submission_df):
    """exception_required must strictly be 0 or 1."""
    assert "exception_required" in submission_df.columns
    vals = submission_df["exception_required"]
    assert vals.isna().sum() == 0, "Found NaNs in exception_required"
    assert set(vals.unique()).issubset({0, 1}), f"Invalid exception_required values: {vals.unique()}"


def test_next_state_valid_classes(submission_df):
    """next_state predictions must belong to the valid state set."""
    assert "next_state" in submission_df.columns
    states = submission_df["next_state"]
    assert states.isna().sum() == 0, "Found NaNs in next_state"
    invalid = set(states.unique()) - set(VALID_STATES)
    assert len(invalid) == 0, f"Found invalid next_state classes: {invalid}"


def test_top_drivers_non_empty(submission_df):
    """top_driver_1, top_driver_2, top_driver_3 must be non-empty strings."""
    for driver_col in ["top_driver_1", "top_driver_2", "top_driver_3"]:
        assert driver_col in submission_df.columns
        assert submission_df[driver_col].isna().sum() == 0, f"Found NaNs in {driver_col}"
        assert (submission_df[driver_col].str.len() > 0).all(), f"Found empty strings in {driver_col}"
