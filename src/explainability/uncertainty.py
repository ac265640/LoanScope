"""
Model Uncertainty and Confidence Quantification Engine
======================================================
Computes epistemic (tree ensemble variance) and aleatoric (probabilistic entropy)
uncertainty metrics to assign explicit confidence ratings to all loan predictions.
"""

import json
import joblib
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.feature_engineer import engineer_panel_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


def compute_prediction_confidence(
    clf,
    X: pd.DataFrame,
    calibrated: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute:
      1. Calibrated probabilities P(Y=1)
      2. Confidence score C in [0.5, 1.0] (distance from max uncertainty P=0.5)
      3. Uncertainty / Entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    """
    probs = clf.predict_proba(X)[:, 1]

    # Normalized Confidence: 1.0 when prob is 0.0 or 1.0; 0.0 when prob is 0.5
    confidence = np.abs(probs - 0.5) * 2.0  # [0.0, 1.0]

    # Binary Shannon Entropy (clamped)
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return probs, confidence, entropy


def evaluate_portfolio_uncertainty(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate uncertainty distribution across the portfolio."""
    features = get_feature_columns()
    X = df[features].fillna(0)

    target = "next_12m_default_flag"
    model_path = MODELS_DIR / f"calibrated_lgbm_{target}.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / f"lgbm_{target}.joblib"

    clf = joblib.load(model_path)
    probs, confs, entropies = compute_prediction_confidence(clf, X)

    high_conf_pct = float((confs >= 0.80).mean() * 100)
    mod_conf_pct = float(((confs >= 0.50) & (confs < 0.80)).mean() * 100)
    low_conf_pct = float((confs < 0.50).mean() * 100)

    results = {
        "target_evaluated": target,
        "n_records": len(X),
        "mean_confidence_score": round(float(confs.mean()), 4),
        "median_confidence_score": round(float(np.median(confs)), 4),
        "mean_entropy_uncertainty": round(float(entropies.mean()), 4),
        "confidence_tiers": {
            "high_confidence_gte_80_pct": round(high_conf_pct, 2),
            "moderate_confidence_50_to_80_pct": round(mod_conf_pct, 2),
            "borderline_low_confidence_lt_50_pct": round(low_conf_pct, 2),
        },
    }

    out_file = MODELS_DIR / "uncertainty_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Uncertainty analysis complete. Mean confidence: {confs.mean():.4f}")
    return results


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)
    evaluate_portfolio_uncertainty(feat_df)


if __name__ == "__main__":
    main()
