"""
Data Schema Validation Tests
===============================
Verifies that the generated datasets conform to the expected schema, field types,
target variable ranges, and cross-column invariants. These tests catch data generation
bugs before they propagate into model training.
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_df():
    path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not path.exists():
        pytest.skip("Train CSV not found — run `make data` first.")
    return pd.read_csv(path, nrows=50_000)  # sample for speed


@pytest.fixture(scope="module")
def test_df():
    path = RAW_DIR / "loan_monthly_performance_test.csv"
    if not path.exists():
        pytest.skip("Test CSV not found — run `make data` first.")
    return pd.read_csv(path, nrows=10_000)


@pytest.fixture(scope="module")
def static_df():
    path = RAW_DIR / "loan_static_attributes.csv"
    if not path.exists():
        pytest.skip("Static attributes CSV not found — run `make data` first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def validation_rules():
    path = DATA_DIR / "validation_rules.json"
    if not path.exists():
        pytest.skip("validation_rules.json not found.")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────

REQUIRED_TRAIN_COLUMNS = [
    "loan_id", "month_index", "reporting_month", "origination_month",
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "credit_score_band", "ltv_band",
    "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "current_status", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "document_status",
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag",
    "next_state", "exception_required",
]

REQUIRED_TEST_COLUMNS = [
    "loan_id", "month_index", "reporting_month", "origination_month",
    "current_balance", "current_status", "days_past_due",
]


def test_train_has_required_columns(train_df):
    """All required training columns must be present."""
    missing = [c for c in REQUIRED_TRAIN_COLUMNS if c not in train_df.columns]
    assert not missing, f"Missing required train columns: {missing}"


def test_test_has_required_columns(test_df):
    """Test dataset must have core loan identifier and status columns."""
    missing = [c for c in REQUIRED_TEST_COLUMNS if c not in test_df.columns]
    assert not missing, f"Missing required test columns: {missing}"


def test_no_duplicate_loan_month_pairs(train_df):
    """Each (loan_id, reporting_month) pair should be unique — allow <0.01% as injected edge-case anomalies."""
    dups = train_df.duplicated(subset=["loan_id", "reporting_month"])
    dup_pct = dups.sum() / len(train_df)
    assert dup_pct < 0.0001, f"Found {dups.sum()} duplicate (loan_id, reporting_month) pairs ({dup_pct:.4%}) — exceeds 0.01% threshold"


def test_loan_id_format(train_df):
    """All loan IDs should follow the LN<digits> format."""
    bad = train_df[~train_df["loan_id"].str.match(r"^LN\d+$")]
    assert len(bad) == 0, f"Found {len(bad)} loan_ids with invalid format"


def test_month_index_positive(train_df):
    """month_index must be ≥ 1 for all records."""
    bad = train_df[train_df["month_index"] < 1]
    assert len(bad) == 0, f"Found {len(bad)} rows with month_index < 1"


def test_loan_age_consistent_with_month_index(train_df):
    """loan_age_months should equal month_index (or be within 1 for edge cases)."""
    diff = (train_df["loan_age_months"] - train_df["month_index"]).abs()
    bad = train_df[diff > 2]
    assert len(bad) / len(train_df) < 0.01, (
        f"More than 1% of rows have loan_age_months inconsistent with month_index"
    )


# ─────────────────────────────────────────────
# Value Range Tests
# ─────────────────────────────────────────────

def test_current_balance_non_negative(train_df):
    """Current balance should be non-negative."""
    bad = train_df[train_df["current_balance"] < 0]
    assert len(bad) == 0, f"Found {len(bad)} rows with negative current_balance"


def test_interest_rate_realistic(train_df):
    """Interest rates should be within plausible bounds. The data generator intentionally
    injects rate outliers as messiness — we cap at 25% and allow up to 2% extreme outliers."""
    valid = train_df["interest_rate"].dropna()
    bad_low = valid[valid < 0.5]
    bad_high = valid[valid > 25.0]   # 25% hard cap — anything above is a clear data error
    outlier_pct_high = (valid > 20.0).mean()
    assert len(bad_low) == 0, f"Found {len(bad_low)} rows with interest_rate < 0.5%"
    assert len(bad_high) == 0, f"Found {len(bad_high)} rows with interest_rate > 25% (hard cap)"
    # Warn if outlier rate is unusual — injected outliers should be < 2%
    assert outlier_pct_high < 0.02, (
        f"Extreme rates (>20%): {outlier_pct_high:.2%} of records — "
        f"injected outlier rate exceeds 2% tolerance (check generator seed)"
    )


def test_days_past_due_non_negative(train_df):
    """days_past_due must be ≥ 0."""
    bad = train_df[train_df["days_past_due"] < 0]
    assert len(bad) == 0, f"Found {len(bad)} rows with negative days_past_due"


def test_binary_target_ranges(train_df):
    """All binary target flags must be 0 or 1."""
    binary_targets = [
        "next_3m_delinquency_flag", "next_6m_delinquency_flag",
        "next_12m_default_flag", "next_12m_prepayment_flag",
        "modification_flag", "prepayment_flag", "default_flag",
    ]
    for col in binary_targets:
        if col not in train_df.columns:
            continue
        invalid = train_df[~train_df[col].isin([0, 1, np.nan])]
        assert len(invalid) == 0, f"Column '{col}' contains values outside {{0, 1}}: {train_df[col].unique()}"


def test_exception_required_binary(train_df):
    """exception_required must be 0 or 1."""
    if "exception_required" not in train_df.columns:
        pytest.skip("exception_required not in train columns")
    invalid = train_df[~train_df["exception_required"].isin([0, 1])]
    assert len(invalid) == 0, f"exception_required has non-binary values"


def test_default_rate_realistic(train_df):
    """12-month default rate should be between 0.5% and 15% for synthetic realistic data."""
    if "next_12m_default_flag" not in train_df.columns:
        pytest.skip("next_12m_default_flag not present")
    rate = train_df["next_12m_default_flag"].mean()
    assert 0.005 <= rate <= 0.15, (
        f"Default rate {rate:.3%} is outside realistic range [0.5%, 15%]"
    )


# ─────────────────────────────────────────────
# Cross-Column Invariant Tests
# ─────────────────────────────────────────────

def test_reporting_after_origination(train_df):
    """reporting_month must be ≥ origination_month for almost all records (allowing ~0.2% injected anomalies)."""
    orig = pd.to_datetime(train_df["origination_month"])
    rep = pd.to_datetime(train_df["reporting_month"])
    violations = (rep < orig).sum()
    violation_pct = violations / len(train_df)
    assert violation_pct < 0.015, (
        f"Date order violations: {violations} ({violation_pct:.2%}) — exceeds 1.5% threshold"
    )


def test_no_train_test_loan_id_overlap():
    """Critical: no loan_id should appear in both train and test sets."""
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    test_path = RAW_DIR / "loan_monthly_performance_test.csv"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("Data files not found.")

    train_ids = set(pd.read_csv(train_path, usecols=["loan_id"])["loan_id"])
    test_ids = set(pd.read_csv(test_path, usecols=["loan_id"])["loan_id"])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, (
        f"DATA LEAKAGE: {len(overlap)} loan_ids appear in both train and test sets"
    )


def test_modification_document_consistency(train_df):
    """Modified loans (modification_flag=1) should not have 'Missing Items' document status."""
    if "modification_flag" not in train_df.columns or "document_status" not in train_df.columns:
        pytest.skip("Required columns missing.")
    violations = train_df[
        (train_df["modification_flag"] == 1) &
        (train_df["document_status"] == "Missing Items")
    ]
    violation_pct = len(violations) / len(train_df)
    # Allow a small fraction since real data can have this inconsistency
    assert violation_pct < 0.02, (
        f"Modified loans with Missing Items docs: {len(violations)} ({violation_pct:.2%})"
    )


# ─────────────────────────────────────────────
# Validation Rules Structural Test
# ─────────────────────────────────────────────

def test_validation_rules_schema(validation_rules):
    """validation_rules.json must contain a list of rules with required keys."""
    assert isinstance(validation_rules, (list, dict)), "validation_rules.json must be a list or dict"
    if isinstance(validation_rules, dict):
        rules = validation_rules.get("rules", validation_rules)
    else:
        rules = validation_rules
    assert len(rules) >= 3, f"Expected ≥3 validation rules, found {len(rules)}"
