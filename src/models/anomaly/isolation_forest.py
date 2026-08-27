"""
Isolation Forest Anomaly Scoring Engine
=======================================
Fits an unsupervised Isolation Forest model on multi-attribute continuous & encoded tabular features.
Generates normalized record-level anomaly scores in [0.0, 1.0] where 1.0 indicates extreme deviation.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.feature_engineer import engineer_panel_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_isolation_forest(
    train_df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> Tuple[IsolationForest, pd.Series]:
    """
    Train Isolation Forest model on tabular panel features.
    
    Justification for Isolation Forest:
    1. Sub-sampling and tree isolation isolates anomalies early near root nodes without assuming Gaussianity.
    2. Efficiently scales to large multidimensional panel datasets (O(n log n)).
    3. Handles non-linear correlations across continuous balances, rates, and discrete risk bands.
    """
    features = get_feature_columns()
    log.info(f"Training Isolation Forest with contamination={contamination} across {len(features)} features...")

    X = train_df[features].fillna(0)

    iso = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)

    # Raw decision function: negative values indicate outliers.
    # We transform to normalized [0.0, 1.0] where 1.0 = most anomalous.
    raw_scores = iso.decision_function(X)
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    normalized_scores = 1.0 - ((raw_scores - min_score) / (max_score - min_score + 1e-8))
    score_series = pd.Series(normalized_scores, index=train_df.index, name="anomaly_score")

    # Save model and normalization constants
    model_path = MODELS_DIR / "isolation_forest.joblib"
    joblib.dump({
        "model": iso,
        "features": features,
        "min_score": float(min_score),
        "max_score": float(max_score),
    }, model_path)

    log.info(
        f"Isolation Forest trained: {len(X):,} records. "
        f"Score range: [{score_series.min():.4f}, {score_series.max():.4f}], "
        f"Mean: {score_series.mean():.4f}, Median: {score_series.median():.4f}"
    )

    return iso, score_series


def predict_anomaly_scores(df: pd.DataFrame) -> np.ndarray:
    """Load model and score any arbitrary panel dataframe."""
    model_path = MODELS_DIR / "isolation_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Isolation Forest model not found at {model_path}")

    artifact = joblib.load(model_path)
    iso = artifact["model"]
    features = artifact["features"]
    min_score = artifact["min_score"]
    max_score = artifact["max_score"]

    X = df[features].fillna(0)
    raw = iso.decision_function(X)
    norm = 1.0 - ((raw - min_score) / (max_score - min_score + 1e-8))
    return np.clip(norm, 0.0, 1.0)


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)
    train_isolation_forest(feat_df)


if __name__ == "__main__":
    main()
