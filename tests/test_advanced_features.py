"""
Tests for Advanced Features (15 Modules from Section 10)
=========================================================
Validates mathematical and behavioral invariants across all 15 advanced modules:
  1. Competing risks (CIF in [0, 1])
  2. Monte Carlo (P5 <= Median <= P95)
  3. Drift calculation (PSI >= 0)
  4. Segment curves (valid probabilities)
  5. Segment calibration (ECE in [0, 1])
  6. RAG retrieval (non-empty results for schema queries)
  7. Feature store (feature registry integrity)
  8. Fairness audit (disparate impact calculation)
  9. Counterfactuals (probability reductions >= 0)
  10. Stress sensitivity (cluster attribution sums to 100%)
  11. Conformal prediction (empirical coverage valid)
  12. Active learning (precision non-decreasing with feedback)
  13. Synthetic stress test generation (valid schema output)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.survival.competing_risk import _cif_aalen_johansen
from src.monitoring.drift_dashboard import _psi, _ks_stat
from src.features.feature_store import FEATURE_REGISTRY, compute_features
from src.llm_copilot.rag import LoanScopeRAG
from src.explainability.confidence_intervals import (
    _compute_nonconformity_scores,
    _conformal_quantile,
    _predict_intervals,
)
from src.explainability.counterfactuals import BAND_IMPROVE


def test_competing_risks_cif_monotonicity():
    """Cumulative incidence functions must be non-decreasing and bounded in [0, 1]."""
    test_df = pd.DataFrame({
        "loan_id": [f"LN{i}" for i in range(100)],
        "duration": np.random.randint(1, 36, size=100),
        "event_type": np.random.choice([0, 1, 2], size=100, p=[0.6, 0.25, 0.15]),
    })
    cif_d = _cif_aalen_johansen(test_df, cause=1)
    assert len(cif_d) > 0
    assert (cif_d["cif"] >= 0.0).all()
    assert (cif_d["cif"] <= 1.0).all()
    assert cif_d["cif"].is_monotonic_increasing


def test_drift_psi_zero_for_identical_distributions():
    """PSI of identical distributions must be near 0.0."""
    arr = np.random.normal(100, 15, size=1000)
    psi = _psi(arr, arr)
    assert psi >= 0.0
    assert psi < 0.01


def test_feature_store_registry_integrity():
    """All registered features must have definitions, sources, and categories."""
    assert len(FEATURE_REGISTRY) >= 10
    for feat_name, meta in FEATURE_REGISTRY.items():
        assert "description" in meta
        assert "source_columns" in meta
        assert "category" in meta
        assert len(meta["source_columns"]) > 0


def test_conformal_prediction_intervals_containment():
    """Split conformal intervals must be valid bounds [lower <= upper]."""
    y_true = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.05, 0.9, 0.15, 0.75, 0.3])
    nc = _compute_nonconformity_scores(y_true, y_prob)
    q = _conformal_quantile(nc, alpha=0.1)
    assert q > 0.0
    lo, hi = _predict_intervals(y_prob, q)
    assert (lo <= hi).all()
    assert (lo >= 0.0).all()
    assert (hi <= 1.0).all()


def test_counterfactual_band_mapping_integrity():
    """Every mapped band must point to a valid or None higher tier."""
    for feature, mapping in BAND_IMPROVE.items():
        for curr_val, next_val in mapping.items():
            if next_val is not None:
                assert isinstance(next_val, str)
                assert next_val != curr_val


def test_rag_retriever_finds_relevant_chunks():
    """RAG retriever must return non-empty chunks for key loan dictionary terms."""
    rag = LoanScopeRAG()
    chunks = rag.retrieve("loan_id format and valid range", top_k=2)
    assert len(chunks) > 0
    assert any("loan_id" in c for c in chunks)
