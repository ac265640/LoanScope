"""
Segment-Level Probability Calibration Audit
============================================
Evaluates probability calibration quality (Brier Score, Expected Calibration Error - ECE)
disaggregated across key portfolio segments:
  1. Credit Score Tiers (<620, 620-659, 660-699, 700-739, 740+)
  2. Origination Vintage Eras (Pre-2010 Legacy, 2010-2019 Post-Crisis, 2020+ Modern)

Detects segment-level overconfidence or underprediction to ensure risk pricing
and loss forecasting are well-calibrated throughout the credit spectrum.

Run: PYTHONPATH=. python src/models/prediction/segment_calibration.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): weighted average gap between confidence and accuracy."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        n_k = np.sum(bin_mask)
        if n_k > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (n_k / total) * abs(bin_acc - bin_conf)

    return float(ece)


def audit_segment_calibration(val_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit calibration per segment."""
    features = get_feature_columns()
    X_val = val_df[features]

    target = "next_12m_default_flag"
    model_path = MODELS_DIR / f"lgbm_{target}.joblib"
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        return {}

    clf = joblib.load(model_path)
    y_true = val_df[target].values
    y_prob = clf.predict_proba(X_val)[:, 1]

    # 1. By Credit Score Band
    credit_results = {}
    for band in val_df["credit_score_band"].dropna().unique():
        mask = (val_df["credit_score_band"] == band).values
        if mask.sum() < 30:
            continue
        g_true, g_prob = y_true[mask], y_prob[mask]
        credit_results[str(band)] = {
            "sample_size": int(mask.sum()),
            "actual_default_rate": round(float(g_true.mean()), 4),
            "mean_predicted_prob": round(float(g_prob.mean()), 4),
            "brier_score": round(float(brier_score_loss(g_true, g_prob)), 4),
            "expected_calibration_error": round(compute_ece(g_true, g_prob), 4),
        }

    # 2. By Vintage Era
    val_df_copy = val_df.copy()
    val_df_copy["orig_year"] = pd.to_datetime(val_df_copy["origination_month"]).dt.year
    val_df_copy["vintage_era"] = pd.cut(
        val_df_copy["orig_year"],
        bins=[1990, 2009, 2019, 2030],
        labels=["Pre-2010 (Legacy)", "2010-2019 (Post-Crisis)", "2020+ (Modern)"]
    )

    vintage_results = {}
    for era in val_df_copy["vintage_era"].dropna().unique():
        mask = (val_df_copy["vintage_era"] == era).values
        if mask.sum() < 30:
            continue
        g_true, g_prob = y_true[mask], y_prob[mask]
        vintage_results[str(era)] = {
            "sample_size": int(mask.sum()),
            "actual_default_rate": round(float(g_true.mean()), 4),
            "mean_predicted_prob": round(float(g_prob.mean()), 4),
            "brier_score": round(float(brier_score_loss(g_true, g_prob)), 4),
            "expected_calibration_error": round(compute_ece(g_true, g_prob), 4),
        }

    results = {
        "overall_brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "overall_ece": round(compute_ece(y_true, y_prob), 4),
        "by_credit_score_band": credit_results,
        "by_vintage_era": vintage_results,
    }

    out_path = MODELS_DIR / "segment_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Segment calibration audit saved → {out_path}")
    return results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error("Train dataset not found.")
        return

    df = pd.read_csv(train_path)
    _, val_df, _ = time_aware_cohort_split(df, val_cutoff="2020-01-01", test_cutoff="2099-01-01")
    val_feat = engineer_panel_features(val_df)
    res = audit_segment_calibration(val_feat)

    print("\n" + "=" * 60)
    print("SEGMENT-LEVEL CALIBRATION REPORT (12M DEFAULT)")
    print("=" * 60)
    print(f"Overall Brier Score: {res['overall_brier_score']:.4f} | Overall ECE: {res['overall_ece']:.4f}")
    print("\nBy Credit Score Band:")
    for b, m in res["by_credit_score_band"].items():
        print(f"  {b:<12} (N={m['sample_size']:>5}) -> Actual: {m['actual_default_rate']:.2%} | Pred: {m['mean_predicted_prob']:.2%} | ECE: {m['expected_calibration_error']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
