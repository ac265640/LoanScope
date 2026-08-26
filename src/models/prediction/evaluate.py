"""
Evaluation & Model Comparison Suite
===================================
Evaluates Baseline vs Improved models with full metric comparison:
  - ROC-AUC, PR-AUC, F1, Recall@Fixed-Precision, Brier Score, Macro-F1
  - Explicit Baseline vs. Improved Deltas
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve, confusion_matrix
)

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


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.60) -> float:
    """Compute maximum recall achievable at or above a specified minimum precision threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_recalls = [r for p, r in zip(precisions, recalls) if p >= target_precision]
    if not valid_recalls:
        return 0.0
    return float(round(max(valid_recalls), 4))


def run_full_evaluation(val_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare baseline and improved LightGBM models on validation set."""
    features = get_feature_columns()
    X_val = val_df[features].fillna(0)

    comparison_report: Dict[str, Any] = {
        "targets": {},
        "summary_table": [],
    }

    # Binary Targets Evaluation
    for target in TARGETS_BINARY:
        y_val = val_df[target].values

        base_path = MODELS_DIR / f"baseline_{target}.joblib"
        lgbm_path = MODELS_DIR / f"calibrated_lgbm_{target}.joblib"
        if not lgbm_path.exists():
            lgbm_path = MODELS_DIR / f"lgbm_{target}.joblib"

        if not base_path.exists() or not lgbm_path.exists():
            log.warning(f"Models for target {target} not found. Skipping evaluation.")
            continue

        base_model = joblib.load(base_path)
        lgbm_model = joblib.load(lgbm_path)

        base_probs = base_model.predict_proba(X_val)[:, 1]
        lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

        base_preds = (base_probs >= 0.5).astype(int)
        lgbm_preds = (lgbm_probs >= 0.5).astype(int)

        # Baseline metrics
        b_auc = float(roc_auc_score(y_val, base_probs))
        b_pr_auc = float(average_precision_score(y_val, base_probs))
        b_brier = float(brier_score_loss(y_val, base_probs))
        b_f1 = float(f1_score(y_val, base_preds, zero_division=0))
        b_rec_p50 = recall_at_precision(y_val, base_probs, target_precision=0.50)

        # Improved LightGBM metrics
        i_auc = float(roc_auc_score(y_val, lgbm_probs))
        i_pr_auc = float(average_precision_score(y_val, lgbm_probs))
        i_brier = float(brier_score_loss(y_val, lgbm_probs))
        i_f1 = float(f1_score(y_val, lgbm_preds, zero_division=0))
        i_rec_p50 = recall_at_precision(y_val, lgbm_probs, target_precision=0.50)

        # Deltas
        d_auc = i_auc - b_auc
        d_pr_auc = i_pr_auc - b_pr_auc
        d_brier = i_brier - b_brier
        d_f1 = i_f1 - b_f1

        comparison_report["targets"][target] = {
            "baseline": {
                "roc_auc": round(b_auc, 4),
                "pr_auc": round(b_pr_auc, 4),
                "brier": round(b_brier, 4),
                "f1": round(b_f1, 4),
                "recall_at_p50": round(b_rec_p50, 4),
            },
            "improved_lgbm": {
                "roc_auc": round(i_auc, 4),
                "pr_auc": round(i_pr_auc, 4),
                "brier": round(i_brier, 4),
                "f1": round(i_f1, 4),
                "recall_at_p50": round(i_rec_p50, 4),
            },
            "deltas": {
                "delta_roc_auc": round(d_auc, 4),
                "delta_pr_auc": round(d_pr_auc, 4),
                "delta_brier": round(d_brier, 4),
                "delta_f1": round(d_f1, 4),
            },
        }

        comparison_report["summary_table"].append({
            "target": target,
            "baseline_auc": round(b_auc, 4),
            "improved_auc": round(i_auc, 4),
            "delta_auc": f"+{d_auc:.4f}" if d_auc >= 0 else f"{d_auc:.4f}",
            "baseline_pr_auc": round(b_pr_auc, 4),
            "improved_pr_auc": round(i_pr_auc, 4),
            "delta_pr_auc": f"+{d_pr_auc:.4f}" if d_pr_auc >= 0 else f"{d_pr_auc:.4f}",
            "baseline_brier": round(b_brier, 4),
            "improved_brier": round(i_brier, 4),
        })

    # Multiclass (next_state)
    base_mc_path = MODELS_DIR / "baseline_next_state.joblib"
    lgbm_mc_path = MODELS_DIR / "lgbm_next_state.joblib"

    if base_mc_path.exists() and lgbm_mc_path.exists():
        y_val_mc = val_df["next_state"].astype(str).values
        base_mc = joblib.load(base_mc_path)
        b_mc_preds = base_mc.predict(X_val)
        b_macro_f1 = float(f1_score(y_val_mc, b_mc_preds, average="macro", zero_division=0))

        lgbm_mc_dict = joblib.load(lgbm_mc_path)
        clf_mc = lgbm_mc_dict["model"]
        idx_to_class = lgbm_mc_dict["idx_to_class"]
        l_idx_preds = clf_mc.predict(X_val)
        l_mc_preds = [idx_to_class[i] for i in l_idx_preds]
        i_macro_f1 = float(f1_score(y_val_mc, l_mc_preds, average="macro", zero_division=0))

        d_macro = i_macro_f1 - b_macro_f1

        comparison_report["targets"]["next_state"] = {
            "baseline_macro_f1": round(b_macro_f1, 4),
            "improved_macro_f1": round(i_macro_f1, 4),
            "delta_macro_f1": round(d_macro, 4),
        }
        comparison_report["summary_table"].append({
            "target": "next_state (multiclass)",
            "baseline_auc": "N/A",
            "improved_auc": "N/A",
            "delta_auc": "N/A",
            "baseline_pr_auc": f"F1: {b_macro_f1:.4f}",
            "improved_pr_auc": f"F1: {i_macro_f1:.4f}",
            "delta_pr_auc": f"ΔF1: {d_macro:+.4f}",
            "baseline_brier": "N/A",
            "improved_brier": "N/A",
        })

    # Save final comparison
    out_file = MODELS_DIR / "model_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(comparison_report, f, indent=2)

    log.info("\n=== Model Performance Comparison Table ===")
    df_sum = pd.DataFrame(comparison_report["summary_table"])
    log.info("\n" + df_sum.to_string(index=False))

    return comparison_report


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error(f"Train data not found at {train_path}.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    _, val_df, _ = time_aware_cohort_split(df, val_cutoff="2020-01-01", test_cutoff="2099-01-01")
    val_feat = engineer_panel_features(val_df)
    run_full_evaluation(val_feat)


if __name__ == "__main__":
    main()
