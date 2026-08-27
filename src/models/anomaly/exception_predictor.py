"""
Hybrid Exception Prediction Engine
==================================
Combines:
  1. Component A (Deterministic Rule Engine): Evaluates hard constraints (VR001-VR005) with 100% precision by construction.
  2. Component B (Learned ML Exception Classifier): Trained strictly on raw/engineered behavioral features (without rule indicator flags) to detect subtle anomalies.
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
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, roc_auc_score

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.models.anomaly.isolation_forest import predict_anomaly_scores
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    signals["total_rule_violations"] = (
        signals["sig_date_anomaly"]
        + signals["sig_paidoff_balance"]
        + signals["sig_default_low_dpd"]
        + signals["sig_mod_missing_doc"]
        + signals["sig_excessive_balance"]
    )
    return signals


def get_rule_derived_exception_type(df: pd.DataFrame) -> pd.Series:
    """Determine deterministic exception type directly from validation rules."""
    signals = compute_rule_violation_signals(df)
    types = pd.Series("None", index=df.index, dtype=object)

    types[signals["sig_date_anomaly"] == 1] = "Date Anomaly"
    types[signals["sig_paidoff_balance"] == 1] = "Balance Inconsistency"
    types[signals["sig_default_low_dpd"] == 1] = "Data Conflict"
    types[signals["sig_mod_missing_doc"] == 1] = "Missing Document"
    types[signals["sig_excessive_balance"] == 1] = "Valuation Discrepancy"

    return types


def train_exception_models(train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Train separate:
      - Component A (Rule Engine Audit): deterministic conformance verification.
      - Component B (Learned ML Model): trained strictly on non-circular behavioral features.
    """
    features = get_feature_columns()
    log.info("Preparing non-circular behavioral feature set for learned ML exception models...")

    # Compute anomaly scores
    anom_scores_train = predict_anomaly_scores(train_df)
    train_df["learned_anomaly_score"] = anom_scores_train

    # Non-circular features: strictly behavioral features + unsupervised anomaly score
    # Note: Rule signals (sig_*) are DELIBERATELY EXCLUDED to avoid circular target leakage
    ml_feature_cols = features + ["learned_anomaly_score"]
    X_train = pd.concat([train_df[features], pd.Series(anom_scores_train, name="learned_anomaly_score", index=train_df.index)], axis=1)

    if val_df is not None:
        anom_scores_val = predict_anomaly_scores(val_df)
        val_df["learned_anomaly_score"] = anom_scores_val
        X_val = pd.concat([val_df[features], pd.Series(anom_scores_val, name="learned_anomaly_score", index=val_df.index)], axis=1)
        y_val_req = val_df["exception_required"].values
        y_val_type_str = val_df["exception_type"].astype(str)
    else:
        X_val = X_train
        y_val_req = train_df["exception_required"].values
        y_val_type_str = train_df["exception_type"].astype(str)

    # 1. Component A: Deterministic Rule Engine Audit
    rule_signals_val = compute_rule_violation_signals(val_df if val_df is not None else train_df)
    rule_pred_req = (rule_signals_val["total_rule_violations"] > 0).astype(int)
    rule_match_rate = float((rule_pred_req == y_val_req).mean())
    log.info(f"Component A (Deterministic Rule Engine) -> Rule Match Rate: {rule_match_rate * 100:.2f}% (by construction)")

    # 2. Component B: Learned ML exception_required Classifier (Binary)
    y_req = train_df["exception_required"].values
    log.info(f"Training non-circular exception_required ML classifier (Positives: {y_req.sum():,} / {len(y_req):,})...")

    scale_weight = max(1.0, float((len(y_req) - y_req.sum()) / max(y_req.sum(), 1)))
    clf_req = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
    )
    clf_req.fit(X_train, y_req)
    req_probs_val = clf_req.predict_proba(X_val)[:, 1]
    req_preds_val = (req_probs_val >= 0.5).astype(int)
    req_f1 = float(f1_score(y_val_req, req_preds_val, zero_division=0))
    req_auc = float(roc_auc_score(y_val_req, req_probs_val))

    log.info(f"Component B (Learned ML Model) exception_required -> ROC-AUC: {req_auc:.4f}, F1 (0.50 cutoff): {req_f1:.4f}")
    req_model_path = MODELS_DIR / "exception_required_model.joblib"
    joblib.dump(clf_req, req_model_path)

    # 3. Component B: Learned ML exception_type Classifier (Multiclass)
    y_type_str = train_df["exception_type"].astype(str)
    unique_types = sorted(y_type_str.unique())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    y_type_idx = y_type_str.map(type_to_idx).values
    y_val_type_idx = y_val_type_str.map(type_to_idx).values

    log.info(f"Training non-circular exception_type ML multiclass classifier across {len(unique_types)} classes...")
    clf_type = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(unique_types),
        n_estimators=150,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_type.fit(X_train, y_type_idx)
    type_preds_val_idx = clf_type.predict(X_val)
    macro_f1_type = float(f1_score(y_val_type_idx, type_preds_val_idx, average="macro", zero_division=0))

    log.info(f"Component B (Learned ML Model) exception_type -> Macro-F1: {macro_f1_type:.4f}")
    type_model_path = MODELS_DIR / "exception_type_model.joblib"
    joblib.dump({
        "model": clf_type,
        "type_to_idx": type_to_idx,
        "idx_to_type": idx_to_type,
        "features": list(X_train.columns),
    }, type_model_path)

    results = {
        "deterministic_rule_engine": {
            "component": "Component A (Deterministic Validation Rules Engine)",
            "rules_evaluated": ["VR001_date_anomaly", "VR002_paidoff_balance", "VR003_default_low_dpd", "VR004_mod_missing_doc", "VR005_excessive_balance_growth"],
            "rule_match_rate": round(rule_match_rate, 4),
            "nature": "Deterministic rule execution (100% conformance by construction on explicit schema/logic violations)",
        },
        "learned_ml_models": {
            "component": "Component B (Learned ML Exception Models - Non-Circular)",
            "features_used": len(ml_feature_cols),
            "features_note": "Trained strictly on raw/engineered behavioral features and unsupervised anomaly score. Rule indicator signals explicitly excluded.",
            "exception_required": {
                "model_type": "Learned LightGBM Classifier",
                "roc_auc": round(req_auc, 4),
                "f1_score_at_05": round(req_f1, 4),
                "model_path": str(req_model_path),
            },
            "exception_type": {
                "model_type": "Learned LightGBM Multiclass Classifier",
                "macro_f1": round(macro_f1_type, 4),
                "classes": unique_types,
                "model_path": str(type_model_path),
            },
        },
        "hybrid_inference_strategy": (
            "Two-stage pipeline: (1) Component A executes deterministic validation rules first, flagging hard constraint "
            "breaches (VR001-VR005) with 100% precision. (2) Component B evaluates remaining records with gradient boosted "
            "trees and Isolation Forest to score probabilistic anomaly and exception risk."
        ),
    }

    out_file = MODELS_DIR / "exception_models_report.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Separated exception models saved to {MODELS_DIR}")
    return results


def predict_hybrid_exceptions(feat_df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Execute transparent two-stage hybrid inference:
      1. Hard deterministic validation rules (VR001-VR005).
      2. Learned ML model on behavioral features for remaining records.
    """
    features = get_feature_columns()
    X = feat_df[features]
    anom_scores = predict_anomaly_scores(feat_df)
    ml_X = pd.concat([X, pd.Series(anom_scores, name="learned_anomaly_score", index=feat_df.index)], axis=1)

    # 1. Evaluate deterministic rule engine
    rule_signals = compute_rule_violation_signals(feat_df)
    rule_types = get_rule_derived_exception_type(feat_df)
    has_rule_violation = (rule_signals["total_rule_violations"] > 0).values

    # 2. Evaluate learned ML model
    req_path = MODELS_DIR / "exception_required_model.joblib"
    if req_path.exists():
        clf_req = joblib.load(req_path)
        ml_pred_req = (clf_req.predict_proba(ml_X)[:, 1] >= 0.5).astype(int)
    else:
        ml_pred_req = np.zeros(len(feat_df), dtype=int)

    type_path = MODELS_DIR / "exception_type_model.joblib"
    if type_path.exists():
        type_dict = joblib.load(type_path)
        clf_type = type_dict["model"]
        idx_to_type = type_dict["idx_to_type"]
        type_preds_idx = clf_type.predict(ml_X)
        ml_pred_types = [idx_to_type[i] for i in type_preds_idx]
    else:
        ml_pred_types = ["None"] * len(feat_df)

    # Combine: Rule engine overrides on hard violations; ML predicts on edge cases
    final_req = np.where(has_rule_violation, 1, ml_pred_req)
    final_type = []
    for i in range(len(feat_df)):
        if has_rule_violation[i]:
            final_type.append(rule_types.iloc[i])
        elif final_req[i] == 1:
            t = ml_pred_types[i]
            final_type.append("Data Conflict" if t in ("None", "nan") else str(t))
        else:
            final_type.append("None")

    return final_req, final_type


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    train_split, val_split, _ = time_aware_cohort_split(df)

    train_feat = engineer_panel_features(train_split)
    val_feat = engineer_panel_features(val_split)

    train_exception_models(train_feat, val_feat)


if __name__ == "__main__":
    main()
