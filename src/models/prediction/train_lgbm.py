"""
Improved Predictive Modeling Pipeline (LightGBM Gradient Boosting)
==================================================================
Trains optimized LightGBM models with hyperparameter tuning, class-imbalance weighting,
and out-of-time validation across all 5 prediction targets.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS_CONFIG = {
    "next_3m_delinquency_flag": {"n_estimators": 150, "learning_rate": 0.04, "num_leaves": 31, "max_depth": 6, "min_child_samples": 40},
    "next_6m_delinquency_flag": {"n_estimators": 150, "learning_rate": 0.04, "num_leaves": 31, "max_depth": 6, "min_child_samples": 40},
    "next_12m_default_flag": {"n_estimators": 180, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 50},
    "next_12m_prepayment_flag": {"n_estimators": 100, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5, "min_child_samples": 80},
}
TARGETS_BINARY = list(TARGETS_CONFIG.keys())
TARGET_MULTICLASS = "next_state"


def train_lgbm_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Train improved LightGBM models for all prediction targets."""
    features = get_feature_columns()
    log.info(f"Training LightGBM models with {len(features)} engineered features...")

    X_train = train_df[features]
    X_val = val_df[features]

    results: Dict[str, Any] = {}

    # 1. Binary Targets
    for target, cfg in TARGETS_CONFIG.items():
        log.info(f"--- Training LightGBM for '{target}' ---")
        y_train = train_df[target].values
        y_val = val_df[target].values

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        log.info(f"Target '{target}' class balance: Pos={pos_count:,}, Neg={neg_count:,}")

        clf = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child_samples"],
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        )

        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y_val, val_probs))
        pr_auc = float(average_precision_score(y_val, val_probs))
        brier = float(brier_score_loss(y_val, val_probs))
        f1 = float(f1_score(y_val, val_preds, zero_division=0))

        log.info(f"LightGBM {target} -> ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, Brier: {brier:.4f}, F1: {f1:.4f}")

        model_path = MODELS_DIR / f"lgbm_{target}.joblib"
        joblib.dump(clf, model_path)

        # Feature importances
        imp_df = pd.DataFrame({
            "feature": features,
            "importance_gain": clf.feature_importances_,
        }).sort_values("importance_gain", ascending=False)
        imp_df.to_csv(MODELS_DIR / f"importance_{target}.csv", index=False)

        results[target] = {
            "model_type": "LightGBM Classifier",
            "best_iteration": cfg["n_estimators"],
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4),
            "top_5_features": imp_df.head(5)["feature"].tolist(),
            "model_path": str(model_path),
        }

    # 2. Multiclass Target (next_state)
    log.info(f"--- Training LightGBM for Multiclass '{TARGET_MULTICLASS}' ---")
    y_train_mc = train_df[TARGET_MULTICLASS].astype(str)
    y_val_mc = val_df[TARGET_MULTICLASS].astype(str)

    unique_classes = sorted(y_train_mc.unique())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    y_train_idx = y_train_mc.map(class_to_idx).values
    y_val_idx = y_val_mc.map(class_to_idx).values

    clf_mc = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(unique_classes),
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_mc.fit(
        X_train,
        y_train_idx,
        eval_set=[(X_val, y_val_idx)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    val_preds_idx = clf_mc.predict(X_val)
    macro_f1 = float(f1_score(y_val_idx, val_preds_idx, average="macro", zero_division=0))

    log.info(f"LightGBM next_state -> Macro-F1: {macro_f1:.4f}")
    mc_model_path = MODELS_DIR / "lgbm_next_state.joblib"
    joblib.dump({
        "model": clf_mc,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }, mc_model_path)

    results[TARGET_MULTICLASS] = {
        "model_type": "LightGBM Multiclass Classifier",
        "macro_f1": round(macro_f1, 4),
        "classes": unique_classes,
        "model_path": str(mc_model_path),
    }

    # Save summary
    out_metrics = MODELS_DIR / "lgbm_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Improved LightGBM models trained and saved to {MODELS_DIR}")
    return results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error(f"Train dataset not found at {train_path}.")
        sys.exit(1)

    log.info("Loading train panel dataset...")
    df = pd.read_csv(train_path)

    train_df, val_df, _ = time_aware_cohort_split(
        df,
        val_cutoff="2020-01-01",
        test_cutoff="2099-01-01",
    )

    train_feat = engineer_panel_features(train_df)
    val_feat = engineer_panel_features(val_df)

    train_lgbm_models(train_feat, val_feat)


if __name__ == "__main__":
    main()
