"""
Global Explainability Engine (SHAP & Permutation Importance)
============================================================
Computes TreeSHAP global feature attributions and ranking across all supervised models.
"""

import sys
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import shap

from src.features.feature_engineer import engineer_panel_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"

TARGETS_BINARY = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def compute_global_shap_importance(df: pd.DataFrame, sample_size: int = 1500) -> Dict[str, Any]:
    """Compute TreeSHAP mean absolute feature importance across targets."""
    features = get_feature_columns()
    sample_df = df[features].dropna().sample(n=min(sample_size, len(df)), random_state=42)

    global_shap_results: Dict[str, Any] = {}

    for target in TARGETS_BINARY:
        model_path = MODELS_DIR / f"lgbm_{target}.joblib"
        if not model_path.exists():
            continue

        log.info(f"Computing TreeSHAP global values for '{target}'...")
        clf = joblib.load(model_path)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample_df)

        # For binary classifier, use positive class SHAP if returned as list
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        feat_ranks = sorted(
            [{"feature": f, "mean_abs_shap": round(float(m), 4)} for f, m in zip(features, mean_abs_shap)],
            key=lambda x: x["mean_abs_shap"],
            reverse=True,
        )

        global_shap_results[target] = {
            "top_10_features": feat_ranks[:10],
            "all_features_shap": feat_ranks,
        }

    out_file = MODELS_DIR / "global_shap_importance.json"
    with open(out_file, "w") as f:
        json.dump(global_shap_results, f, indent=2)

    log.info(f"✅ Global SHAP feature importances saved to {out_file}")
    return global_shap_results


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)
    compute_global_shap_importance(feat_df)


if __name__ == "__main__":
    main()
