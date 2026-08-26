"""
Probability Calibration Engine
==============================
Fits Platt Scaling (Sigmoid) & Isotonic Calibration to ensure predicted probabilities
reflect empirical risk frequencies. Generates Expected Calibration Error (ECE) metrics.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"

TARGETS_BINARY = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE):
      ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def calibrate_prediction_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Fit isotonic/sigmoid calibration on validation set for binary classifiers."""
    features = get_feature_columns()
    X_train = train_df[features]
    X_val = val_df[features]

    calibration_report: Dict[str, Any] = {}

    for target in TARGETS_BINARY:
        model_path = MODELS_DIR / f"lgbm_{target}.joblib"
        if not model_path.exists():
            log.warning(f"Model for {target} not found at {model_path}. Skipping.")
            continue

        clf = joblib.load(model_path)
        y_train = train_df[target].values
        y_val = val_df[target].values

        # Raw probabilities
        raw_val_probs = clf.predict_proba(X_val)[:, 1]
        raw_brier = float(brier_score_loss(y_val, raw_val_probs))
        raw_ece = calculate_ece(y_val, raw_val_probs)

        # Calibrated wrapper
        log.info(f"Calibrating '{target}' probabilities with Platt / Isotonic scaling...")
        calibrated_clf = CalibratedClassifierCV(
            estimator=clf,
            method="sigmoid",
            cv="prefit",
        )
        calibrated_clf.fit(X_val, y_val)

        cal_probs = calibrated_clf.predict_proba(X_val)[:, 1]
        cal_brier = float(brier_score_loss(y_val, cal_probs))
        cal_ece = calculate_ece(y_val, cal_probs)

        # Reliability curve points
        prob_true, prob_pred = calibration_curve(y_val, cal_probs, n_bins=10)
        curve_points = [
            {"pred_bin": round(float(p), 4), "actual_empirical": round(float(t), 4)}
            for p, t in zip(prob_pred, prob_true)
        ]

        # Save calibrated model
        cal_model_path = MODELS_DIR / f"calibrated_lgbm_{target}.joblib"
        joblib.dump(calibrated_clf, cal_model_path)

        log.info(
            f"Target {target} -> "
            f"Brier: {raw_brier:.4f} -> {cal_brier:.4f} (Δ={cal_brier - raw_brier:.4f}), "
            f"ECE: {raw_ece:.4f} -> {cal_ece:.4f}"
        )

        calibration_report[target] = {
            "uncalibrated_brier": round(raw_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "uncalibrated_ece": round(raw_ece, 4),
            "calibrated_ece": round(cal_ece, 4),
            "calibration_method": "Platt Scaling (Sigmoid)",
            "reliability_curve": curve_points,
            "calibrated_model_path": str(cal_model_path),
        }

    report_path = MODELS_DIR / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(calibration_report, f, indent=2)

    log.info(f"✅ Calibration completed and saved to {report_path}")
    return calibration_report


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error(f"Train data not found at {train_path}.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    train_df, val_df, _ = time_aware_cohort_split(df, val_cutoff="2020-01-01", test_cutoff="2099-01-01")

    train_feat = engineer_panel_features(train_df)
    val_feat = engineer_panel_features(val_df)

    calibrate_prediction_models(train_feat, val_feat)


if __name__ == "__main__":
    main()
