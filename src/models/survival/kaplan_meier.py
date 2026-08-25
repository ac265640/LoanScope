"""
Kaplan-Meier Survival Analysis Module
=====================================
Estimates non-parametric survival and cumulative default hazard curves across loan cohorts
segmented by Credit Score Band and Origination Vintage. Handles right-censoring cleanly.
"""

from typing import Dict, Any, List
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "reports"


def prepare_survival_dataset(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate loan panel to loan-level time-to-event structure:
      - duration (T): total observation months until default or right-censoring
      - event (E): 1 if loan defaulted, 0 if censored (prepaid, active, paid off)
    """
    log.info("Aggregating panel to loan-level duration and event status...")
    sorted_df = panel_df.sort_values(["loan_id", "month_index"])

    loan_events = sorted_df.groupby("loan_id").agg({
        "month_index": "max",
        "default_flag": "max",
        "credit_score_band": "first",
        "origination_month": "first",
        "state": "first",
        "interest_rate": "first",
        "original_balance": "first",
    }).reset_index()

    loan_events.rename(columns={"month_index": "duration", "default_flag": "event"}, inplace=True)
    loan_events["orig_year"] = pd.to_datetime(loan_events["origination_month"] + "-01", errors="coerce").dt.year.fillna(2018).astype(int)
    loan_events["credit_score_band"] = loan_events["credit_score_band"].fillna("MISSING")

    return loan_events


def fit_kaplan_meier_curves(loan_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit overall and segmented Kaplan-Meier estimators.
    """
    kmf = KaplanMeierFitter()
    results: Dict[str, Any] = {
        "overall_survival_table": [],
        "by_credit_band": {},
        "by_vintage": {},
    }

    # 1. Overall survival
    kmf.fit(loan_df["duration"], event_observed=loan_df["event"], label="Overall Portfolio")
    surv_table = kmf.survival_function_.reset_index()
    surv_table.columns = ["timeline_months", "survival_prob"]
    surv_table["cumulative_default_prob"] = 1.0 - surv_table["survival_prob"]
    results["overall_survival_table"] = surv_table.round(4).to_dict(orient="records")

    # 2. Segmented by Credit Score Band
    for band in ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]:
        sub = loan_df[loan_df["credit_score_band"] == band]
        if len(sub) > 20:
            kmf_band = KaplanMeierFitter()
            kmf_band.fit(sub["duration"], event_observed=sub["event"], label=f"Credit {band}")
            cum_def = (1.0 - kmf_band.survival_function_).reset_index()
            cum_def.columns = ["timeline_months", "cum_default_rate"]
            results["by_credit_band"][band] = {
                "n_loans": len(sub),
                "observed_defaults": int(sub["event"].sum()),
                "default_rate_pct": round(float(sub["event"].mean() * 100), 2),
                "12m_cumulative_default": round(float(cum_def[cum_def["timeline_months"] <= 12]["cum_default_rate"].iloc[-1]), 4) if len(cum_def[cum_def["timeline_months"] <= 12]) > 0 else 0.0,
                "24m_cumulative_default": round(float(cum_def[cum_def["timeline_months"] <= 24]["cum_default_rate"].iloc[-1]), 4) if len(cum_def[cum_def["timeline_months"] <= 24]) > 0 else 0.0,
            }

    # 3. Segmented by Vintage Era
    loan_df["vintage_era"] = np.where(
        loan_df["orig_year"] <= 2010, "Crisis/Pre-2010",
        np.where(loan_df["orig_year"] <= 2018, "Post-Crisis 2011-2018", "Recent 2019+")
    )
    for era in ["Crisis/Pre-2010", "Post-Crisis 2011-2018", "Recent 2019+"]:
        sub = loan_df[loan_df["vintage_era"] == era]
        if len(sub) > 20:
            kmf_era = KaplanMeierFitter()
            kmf_era.fit(sub["duration"], event_observed=sub["event"], label=era)
            cum_def = (1.0 - kmf_era.survival_function_).reset_index()
            cum_def.columns = ["timeline_months", "cum_default_rate"]
            results["by_vintage"][era] = {
                "n_loans": len(sub),
                "observed_defaults": int(sub["event"].sum()),
                "default_rate_pct": round(float(sub["event"].mean() * 100), 2),
                "12m_cumulative_default": round(float(cum_def[cum_def["timeline_months"] <= 12]["cum_default_rate"].iloc[-1]), 4) if len(cum_def[cum_def["timeline_months"] <= 12]) > 0 else 0.0,
            }

    return results


def main():
    raw_path = ROOT / "data" / "raw" / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    surv_df = prepare_survival_dataset(df)
    results = fit_kaplan_meier_curves(surv_df)

    out_json = ROOT / "src" / "models" / "saved_models" / "kaplan_meier_results.json"
    import json
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Kaplan-Meier survival curves fitted and saved to {out_json}")


if __name__ == "__main__":
    main()
