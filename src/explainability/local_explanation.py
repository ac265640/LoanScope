"""
Local Explainability Engine (SHAP Waterfall Attributions per Loan)
==================================================================
Computes instance-level local feature contributions (base value + feature impacts = predicted log-odds).
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


def explain_single_loan(
    loan_record: pd.Series,
    target: str = "next_12m_default_flag",
) -> Dict[str, Any]:
    """Generate detailed local SHAP attribution waterfall for an individual loan."""
    model_path = MODELS_DIR / f"lgbm_{target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model for {target} not found at {model_path}")

    clf = joblib.load(model_path)
    features = get_feature_columns()

    row_features = pd.DataFrame([loan_record[features].fillna(0)])
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(row_features)

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]

    base_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

    attributions = []
    for f_name, s_val, f_raw in zip(features, sv, row_features.iloc[0]):
        attributions.append({
            "feature": f_name,
            "feature_value": round(float(f_raw), 4) if isinstance(f_raw, (int, float, np.number)) else str(f_raw),
            "shap_attribution": round(float(s_val), 4),
            "direction": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK",
        })

    # Sort by absolute impact
    attributions = sorted(attributions, key=lambda x: abs(x["shap_attribution"]), reverse=True)

    prob = float(clf.predict_proba(row_features)[0, 1])

    return {
        "loan_id": str(loan_record.get("loan_id", "UNKNOWN")),
        "reporting_month": str(loan_record.get("reporting_month", "UNKNOWN")),
        "target": target,
        "predicted_probability": round(prob, 4),
        "base_expected_value": round(base_val, 4),
        "top_positive_risk_drivers": [a for a in attributions if a["direction"] == "INCREASES_RISK"][:5],
        "top_negative_protective_drivers": [a for a in attributions if a["direction"] == "DECREASES_RISK"][:5],
        "all_attributions": attributions,
    }


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    feat_df = engineer_panel_features(df)

    # Pick 3 representative loans: 1 high risk, 1 medium risk, 1 prime
    sample_loans = feat_df.sample(n=3, random_state=42)
    local_examples = []
    for _, row in sample_loans.iterrows():
        exp = explain_single_loan(row, target="next_12m_default_flag")
        local_examples.append(exp)

    out_file = MODELS_DIR / "local_shap_examples.json"
    with open(out_file, "w") as f:
        json.dump(local_examples, f, indent=2)

    log.info(f"✅ Local SHAP explanations generated -> saved to {out_file}")


if __name__ == "__main__":
    main()
