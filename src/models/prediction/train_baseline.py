"""
Baseline Model Training Pipeline (Logistic Regression & Simple Baselines)
========================================================================
Trains standard regularized Logistic Regression models across all 5 prediction targets:
  - next_3m_delinquency_flag
  - next_6m_delinquency_flag
  - next_12m_default_flag
  - next_12m_prepayment_flag
  - next_state (multiclass LogisticRegression)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


TARGETS_BINARY = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]
TARGET_MULTICLASS = "next_state"


def train_baseline_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Train baseline Logistic Regression models for all targets."""
    features = get_feature_columns()
    log.info(f"Using {len(features)} features for baseline modeling.")

    X_train = train_df[features].fillna(0)
    X_val = val_df[features].fillna(0)

    results: Dict[str, Any] = {}

    # 1. Binary targets
    for target in TARGETS_BINARY:
        log.info(f"Training Baseline Logistic Regression for '{target}'...")
        y_train = train_df[target].values
        y_val = val_df[target].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=42,
                C=1.0,
            )),
        ])
        pipe.fit(X_train, y_train)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        val_preds = pipe.predict(X_val)

        auc = float(roc_auc_score(y_val, val_probs))
        pr_auc = float(average_precision_score(y_val, val_probs))
        brier = float(brier_score_loss(y_val, val_probs))
        f1 = float(f1_score(y_val, val_preds, zero_division=0))

        log.info(f"Baseline {target} -> ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, Brier: {brier:.4f}, F1: {f1:.4f}")

        model_path = MODELS_DIR / f"baseline_{target}.joblib"
        joblib.dump(pipe, model_path)

        results[target] = {
            "model_type": "LogisticRegression (StandardScaled, Balanced)",
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4),
            "model_path": str(model_path),
        }

    # 2. Multiclass target (next_state)
    log.info(f"Training Baseline Logistic Regression for multiclass '{TARGET_MULTICLASS}'...")
    y_train_mc = train_df[TARGET_MULTICLASS].astype(str).values
    y_val_mc = val_df[TARGET_MULTICLASS].astype(str).values

    pipe_mc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )),
    ])
    pipe_mc.fit(X_train, y_train_mc)
    val_preds_mc = pipe_mc.predict(X_val)
    macro_f1 = float(f1_score(y_val_mc, val_preds_mc, average="macro", zero_division=0))

    log.info(f"Baseline next_state -> Macro-F1: {macro_f1:.4f}")
    mc_path = MODELS_DIR / "baseline_next_state.joblib"
    joblib.dump(pipe_mc, mc_path)

    results[TARGET_MULTICLASS] = {
        "model_type": "Multinomial LogisticRegression",
        "macro_f1": round(macro_f1, 4),
        "model_path": str(mc_path),
    }

    # Save metrics summary
    metrics_path = MODELS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        import json
        json.dump(results, f, indent=2)

    log.info(f"✅ Baseline models trained and saved to {MODELS_DIR}")
    return results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error(f"Train dataset not found at {train_path}.")
        sys.exit(1)

    log.info("Loading train panel dataset...")
    df = pd.read_csv(train_path)

    # Time-aware split within train panel (e.g. split into Train & Validation cohorts)
    train_df, val_df, _ = time_aware_cohort_split(
        df,
        val_cutoff="2020-01-01",
        test_cutoff="2099-01-01",
    )

    train_feat = engineer_panel_features(train_df)
    val_feat = engineer_panel_features(val_df)

    train_baseline_models(train_feat, val_feat)


if __name__ == "__main__":
    main()
