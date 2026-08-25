"""
Pytest Fixtures and Global Test Configuration
=============================================
Provides shared test fixtures, deterministic random seed settings,
and mock datasets for the Loan Performance Intelligence Engine test suite.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def set_global_random_seed():
    """Set numpy seed for reproducible test executions."""
    np.random.seed(42)


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def sample_loan_feature_vector():
    """Returns a realistic single-row dictionary of engineered features."""
    return {
        "loan_age_months": 18,
        "remaining_term_months": 342,
        "original_balance": 350000.0,
        "current_balance": 338500.0,
        "interest_rate_imputed": 5.25,
        "interest_rate_is_missing": 0,
        "credit_score_is_missing": 0,
        "credit_score_ordinal": 4,
        "ltv_ordinal": 3,
        "dti_ordinal": 3,
        "current_status_code": 1,
        "days_past_due": 30,
        "modification_flag": 0,
        "balance_to_orig_ratio": 0.967,
        "rate_to_market_spread": 1.25,
        "seasoning_ratio": 0.05,
        "dpd_roll_max_3m": 30.0,
        "dpd_roll_max_6m": 30.0,
        "dpd_roll_mean_6m": 5.0,
        "balance_change_1m_pct": -0.002,
        "balance_change_3m_pct": -0.006,
        "is_currently_delinquent": 1,
        "ever_delinquent_past": 1,
        "state_freq": 0.12,
        "servicer_name_freq": 0.25,
        "loan_purpose_freq": 0.65,
        "occupancy_type_freq": 0.85,
        "property_type_freq": 0.70,
        "document_status_freq": 0.90,
        "source_system_freq": 0.75,
        "orig_year": 2021,
        "is_legacy_vintage": 0,
    }
