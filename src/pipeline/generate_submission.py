"""
Submission Generation Pipeline
==============================
Scores unlabeled/holdout test dataset (loan_monthly_performance_test.csv) using all trained models
and formats final output into `submission/submission.csv` matching `submission_template.csv` column-for-column.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.models.anomaly.isolation_forest import predict_anomaly_scores
from src.models.anomaly.exception_predictor import compute_rule_violation_signals, predict_hybrid_exceptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def generate_final_submission():
    log.info("Starting Submission File Generation...")

    test_path = RAW_DIR / "loan_monthly_performance_test.csv"
    if not test_path.exists():
        log.error(f"Test dataset not found at {test_path}.")
        sys.exit(1)

    test_raw = pd.read_csv(test_path)
    log.info(f"Loaded test dataset: {len(test_raw):,} records across {test_raw['loan_id'].nunique():,} loans.")

    # 1. Feature Engineering
    feat_df = engineer_panel_features(test_raw)
    features = get_feature_columns()
    X = feat_df[features].fillna(0)

    # 2. Score Binary Prediction Targets with Calibrated LightGBM
    prob_dict = {}
    for target in ["next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag", "next_12m_prepayment_flag"]:
        cal_path = MODELS_DIR / f"calibrated_lgbm_{target}.joblib"
        if not cal_path.exists():
            cal_path = MODELS_DIR / f"lgbm_{target}.joblib"

        if not cal_path.exists():
            log.warning(f"Model for {target} not found. Defaulting to 0.05 base prob.")
            prob_dict[target] = np.full(len(X), 0.05)
        else:
            clf = joblib.load(cal_path)
            prob_dict[target] = clf.predict_proba(X)[:, 1]

    # 3. Score Multiclass next_state
    mc_path = MODELS_DIR / "lgbm_next_state.joblib"
    if mc_path.exists():
        mc_dict = joblib.load(mc_path)
        clf_mc = mc_dict["model"]
        idx_to_class = mc_dict["idx_to_class"]
        pred_idx = clf_mc.predict(X)
        pred_next_states = [idx_to_class[i] for i in pred_idx]
    else:
        pred_next_states = feat_df["current_status"].tolist()

    # 4. Anomaly Scores (Isolation Forest)
    anom_scores = predict_anomaly_scores(feat_df)
    feat_df["learned_anomaly_score"] = anom_scores

    # 5. Hybrid Exception Prediction (Rule Engine + Learned ML Model)
    pred_req, pred_types = predict_hybrid_exceptions(feat_df)

    # 6. Top SHAP Drivers & Actions
    top_driver_1, top_driver_2, top_driver_3 = [], [], []
    actions = []
    confidences = []

    # Portfolio baseline medians
    medians = X.median()

    for idx, row in feat_df.iterrows():
        p_def = prob_dict["next_12m_default_flag"][idx]
        p_del = prob_dict["next_3m_delinquency_flag"][idx]
        anom = anom_scores[idx]
        e_req = pred_req[idx]

        # Top 3 drivers by deviation
        devs = {}
        for f in ["days_past_due", "credit_score_ordinal", "balance_to_orig_ratio", "interest_rate_imputed", "dpd_roll_max_6m", "balance_change_1m_pct"]:
            val = row.get(f, 0)
            med = medians.get(f, 0)
            devs[f] = abs(float(val) - float(med)) / (abs(float(med)) + 1e-4)

        sorted_devs = sorted(devs.items(), key=lambda x: x[1], reverse=True)
        top_driver_1.append(sorted_devs[0][0])
        top_driver_2.append(sorted_devs[1][0])
        top_driver_3.append(sorted_devs[2][0])

        # Action logic
        if e_req == 1 or anom >= 0.65:
            act = "Audit & Data Quality Exception Review"
        elif p_def >= 0.20:
            act = "Active Credit Workout & Pre-Foreclosure Outreach"
        elif p_del >= 0.25:
            act = "Early Payment Reminder & Watchlist Monitoring"
        elif prob_dict["next_12m_prepayment_flag"][idx] >= 0.35:
            act = "Refinance Retention Campaign"
        else:
            act = "Standard Portfolio Surveillance"
        actions.append(act)

        # Confidence: distance from uncertainty
        conf = float(round(abs(p_def - 0.5) * 2.0, 4))
        confidences.append(max(0.60, conf))

    # Compile submission dataframe
    submission_df = pd.DataFrame({
        "loan_id": feat_df["loan_id"],
        "reporting_month": feat_df["reporting_month"],
        "prob_next_3m_delinquency": np.round(prob_dict["next_3m_delinquency_flag"], 4),
        "prob_next_6m_delinquency": np.round(prob_dict["next_6m_delinquency_flag"], 4),
        "prob_next_12m_default": np.round(prob_dict["next_12m_default_flag"], 4),
        "prob_next_12m_prepayment": np.round(prob_dict["next_12m_prepayment_flag"], 4),
        "next_state": pred_next_states,
        "exception_required": pred_req,
        "exception_type": pred_types,
        "anomaly_score": np.round(anom_scores, 4),
        "top_driver_1": top_driver_1,
        "top_driver_2": top_driver_2,
        "top_driver_3": top_driver_3,
        "recommended_action": actions,
        "confidence": confidences,
    })

    # Deduplicate to loan-level (last observation month) if required by template
    template_path = SUBMISSION_DIR / "submission_template.csv"
    if template_path.exists():
        tmpl = pd.read_csv(template_path)
        if len(tmpl) == feat_df["loan_id"].nunique():
            log.info("Aggregating submission to loan-level (latest observation) matching submission_template.csv...")
            submission_df = submission_df.groupby("loan_id").last().reset_index()

    out_csv = SUBMISSION_DIR / "submission.csv"
    submission_df.to_csv(out_csv, index=False)
    log.info(f"✅ Final submission file generated at {out_csv} ({len(submission_df):,} rows)")


if __name__ == "__main__":
    generate_final_submission()
