"""
Classification Threshold Optimizer
=====================================
Finds the optimal probability threshold per binary target that maximises
F1-score and analyses the precision-recall tradeoff at multiple cutoffs.

For imbalanced credit-risk targets the default 0.5 threshold is almost never
optimal; this module provides production-ready cutoffs tuned on the validation
cohort and records the precision / recall / F1 curve.

Run: PYTHONPATH=. python src/models/prediction/threshold_optimizer.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"

TARGETS_BINARY = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]

CANDIDATE_THRESHOLDS = np.round(np.arange(0.05, 0.75, 0.05), 2).tolist()


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Sweep candidate thresholds and return the one that maximises F1-score,
    along with the full precision-recall curve for reporting.
    """
    rows = []
    for t in CANDIDATE_THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})

    curve = pd.DataFrame(rows)
    best_row = curve.loc[curve["f1"].idxmax()]

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_precision": float(best_row["precision"]),
        "optimal_recall": float(best_row["recall"]),
        "optimal_f1": float(best_row["f1"]),
        "default_05_f1": float(curve.loc[curve["threshold"] == 0.5, "f1"].values[0])
        if 0.5 in curve["threshold"].values else None,
        "pr_curve": curve.to_dict(orient="records"),
    }


def run_threshold_analysis(val_df: pd.DataFrame) -> dict:
    """Run threshold optimization across all binary targets."""
    features = get_feature_columns()
    X_val = val_df[features]
    results = {}

    for target in TARGETS_BINARY:
        model_path = MODELS_DIR / f"lgbm_{target}.joblib"
        if not model_path.exists():
            log.warning(f"Model not found for {target}. Skipping.")
            continue

        log.info(f"Optimising threshold for '{target}'...")
        clf = joblib.load(model_path)
        y_val = val_df[target].values
        y_prob = clf.predict_proba(X_val)[:, 1]

        threshold_result = find_optimal_threshold(y_val, y_prob)
        threshold_result["roc_auc"] = round(float(roc_auc_score(y_val, y_prob)), 4)
        threshold_result["positive_rate"] = round(float(y_val.mean()), 4)

        log.info(
            f"  {target}: optimal_threshold={threshold_result['optimal_threshold']:.2f} "
            f"F1={threshold_result['optimal_f1']:.4f} "
            f"(vs default 0.5 F1={threshold_result['default_05_f1']})"
        )
        results[target] = threshold_result

    out_path = MODELS_DIR / "threshold_optimization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"✅ Threshold optimization results saved → {out_path}")
    return results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error("Train dataset not found. Run `make data` first.")
        sys.exit(1)

    log.info("Loading validation cohort for threshold analysis...")
    df = pd.read_csv(train_path)
    _, val_df, _ = time_aware_cohort_split(df, val_cutoff="2020-01-01", test_cutoff="2099-01-01")
    val_feat = engineer_panel_features(val_df)
    run_threshold_analysis(val_feat)


if __name__ == "__main__":
    main()
