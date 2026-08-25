"""
Hybrid Exception Prediction Engine
==================================
Combines deterministic validation rule violations with learned unsupervised anomaly scores
to accurately predict `exception_required` (binary) and `exception_type` (multiclass).
"""

import sys
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, roc_auc_score

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.models.anomaly.isolation_forest import predict_anomaly_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


def compute_rule_violation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute explicit binary indicators for starter validation rules from validation_rules.json."""
    signals = pd.DataFrame(index=df.index)

    # VR001: reporting_month < origination_month
    if "reporting_month" in df.columns and "origination_month" in df.columns:
        r_dt = pd.to_datetime(df["reporting_month"] + "-01", errors="coerce")
        o_dt = pd.to_datetime(df["origination_month"] + "-01", errors="coerce")
        signals["sig_date_anomaly"] = (r_dt < o_dt).fillna(False).astype(int)
    else:
        signals["sig_date_anomaly"] = 0

    # VR002: Paid Off with positive balance (>1000)
    if "current_status" in df.columns and "current_balance" in df.columns:
        signals["sig_paidoff_balance"] = (
            (df["current_status"] == "Paid Off") & (df["current_balance"] > 1000)
        ).fillna(False).astype(int)
    else:
        signals["sig_paidoff_balance"] = 0

    # VR003: Default with DPD < 60
    if "current_status" in df.columns and "days_past_due" in df.columns:
        signals["sig_default_low_dpd"] = (
            (df["current_status"] == "Default") & (df["days_past_due"] < 60)
        ).fillna(False).astype(int)
    else:
        signals["sig_default_low_dpd"] = 0

    # VR004: Missing document with modification
    if "modification_flag" in df.columns and "document_status" in df.columns:
        signals["sig_mod_missing_doc"] = (
            (df["modification_flag"] == 1) & (df["document_status"] == "Missing Items")
        ).fillna(False).astype(int)
    else:
        signals["sig_mod_missing_doc"] = 0

    # VR005: Excessive balance growth (> 2x original)
    if "current_balance" in df.columns and "original_balance" in df.columns:
        signals["sig_excessive_balance"] = (
            df["current_balance"] > df["original_balance"] * 2.0
        ).fillna(False).astype(int)
    else:
        signals["sig_excessive_balance"] = 0

    # Composite rule violation count
    signals["total_rule_violations"] = signals.sum(axis=1)
    return signals


def train_exception_models(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Train hybrid model for exception_required and exception_type."""
    features = get_feature_columns()
    log.info("Generating rule features and unsupervised anomaly scores...")

    rule_df = compute_rule_violation_signals(train_df)
    anom_scores = predict_anomaly_scores(train_df)
    train_df["learned_anomaly_score"] = anom_scores

    hybrid_feature_cols = features + list(rule_df.columns) + ["learned_anomaly_score"]
    X = pd.concat([train_df[features], rule_df, pd.Series(anom_scores, name="learned_anomaly_score", index=train_df.index)], axis=1)

    # 1. Predict exception_required (Binary)
    y_req = train_df["exception_required"].values
    log.info(f"Training exception_required classifier (Positive cases: {y_req.sum():,} / {len(y_req):,})...")

    scale_weight = max(1.0, float((len(y_req) - y_req.sum()) / max(y_req.sum(), 1)))
    clf_req = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
    )
    clf_req.fit(X, y_req)
    req_probs = clf_req.predict_proba(X)[:, 1]
    req_preds = (req_probs >= 0.5).astype(int)
    req_f1 = float(f1_score(y_req, req_preds, zero_division=0))
    req_auc = float(roc_auc_score(y_req, req_probs))

    log.info(f"Exception Required Model -> F1: {req_f1:.4f}, ROC-AUC: {req_auc:.4f}")
    req_model_path = MODELS_DIR / "exception_required_model.joblib"
    joblib.dump(clf_req, req_model_path)

    # 2. Predict exception_type (Multiclass)
    y_type_str = train_df["exception_type"].astype(str)
    unique_types = sorted(y_type_str.unique())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    y_type_idx = y_type_str.map(type_to_idx).values

    log.info(f"Training exception_type multiclass classifier across {len(unique_types)} classes...")
    clf_type = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(unique_types),
        n_estimators=150,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_type.fit(X, y_type_idx)
    type_preds_idx = clf_type.predict(X)
    macro_f1_type = float(f1_score(y_type_idx, type_preds_idx, average="macro", zero_division=0))

    log.info(f"Exception Type Model -> Macro-F1: {macro_f1_type:.4f}")
    type_model_path = MODELS_DIR / "exception_type_model.joblib"
    joblib.dump({
        "model": clf_type,
        "type_to_idx": type_to_idx,
        "idx_to_type": idx_to_type,
        "features": list(X.columns),
    }, type_model_path)

    results = {
        "exception_required": {
            "model_type": "Hybrid LightGBM Classifier",
            "f1_score": round(req_f1, 4),
            "roc_auc": round(req_auc, 4),
            "model_path": str(req_model_path),
        },
        "exception_type": {
            "model_type": "Hybrid LightGBM Multiclass Classifier",
            "macro_f1": round(macro_f1_type, 4),
            "classes": unique_types,
            "model_path": str(type_model_path),
        },
    }

    out_file = MODELS_DIR / "exception_models_report.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Exception prediction models saved to {MODELS_DIR}")
    return results


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)
    train_exception_models(feat_df)


if __name__ == "__main__":
    main()
