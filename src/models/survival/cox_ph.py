"""
Cox Proportional Hazards Survival Model
=======================================
Fits semi-parametric Cox Proportional Hazards regression to quantify hazard ratios
and survival trajectories conditional on credit score, LTV, DTI, interest rate, and vintage.
"""

import sys
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.models.survival.kaplan_meier import prepare_survival_dataset
from src.features.feature_engineer import CREDIT_BAND_MAP, LTV_BAND_MAP, DTI_BAND_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


def fit_cox_model(surv_df: pd.DataFrame, full_panel: pd.DataFrame) -> Dict[str, Any]:
    """Fit Cox Proportional Hazards model on loan-level survival data."""
    log.info("Preparing covariates for Cox Proportional Hazards model...")

    # Merge static underwriting covariates
    static_cols = ["loan_id", "ltv_band", "dti_band"]
    statics = full_panel.drop_duplicates(subset=["loan_id"])[static_cols]
    m_df = pd.merge(surv_df, statics, on="loan_id", how="left")

    m_df["credit_score_ordinal"] = m_df["credit_score_band"].map(CREDIT_BAND_MAP).fillna(3).astype(float)
    m_df["ltv_ordinal"] = m_df["ltv_band"].map(LTV_BAND_MAP).fillna(3).astype(float)
    m_df["dti_ordinal"] = m_df["dti_band"].map(DTI_BAND_MAP).fillna(3).astype(float)
    m_df["interest_rate"] = m_df["interest_rate"].fillna(m_df["interest_rate"].median())
    m_df["is_legacy_vintage"] = (m_df["orig_year"] < 2012).astype(float)

    cox_features = [
        "duration", "event",
        "credit_score_ordinal", "ltv_ordinal", "dti_ordinal",
        "interest_rate", "is_legacy_vintage",
    ]
    train_cox = m_df[cox_features].dropna()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train_cox, duration_col="duration", event_col="event")

    c_index = float(cph.concordance_index_)
    log.info(f"Cox PH Model fitted -> Concordance Index (C-Index): {c_index:.4f}")

    # Extract hazard ratios
    hr_summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].reset_index()
    hr_summary.columns = ["covariate", "coef", "hazard_ratio", "se", "p_value"]
    hr_list = hr_summary.round(4).to_dict(orient="records")

    # Save model
    model_path = MODELS_DIR / "cox_ph_model.joblib"
    joblib.dump(cph, model_path)

    results = {
        "model_type": "Cox Proportional Hazards (L2 Penalized)",
        "concordance_index": round(c_index, 4),
        "hazard_ratios": hr_list,
        "n_observations": len(train_cox),
        "n_events": int(train_cox["event"].sum()),
    }

    out_json = MODELS_DIR / "cox_ph_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Cox PH model saved to {model_path}")
    return results


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    surv_df = prepare_survival_dataset(df)
    fit_cox_model(surv_df, df)


if __name__ == "__main__":
    main()
