"""
Competing-Risk Survival Modeling (Default vs. Prepayment)
=========================================================
Estimates cause-specific hazards and Cumulative Incidence Functions (CIF)
for mutually exclusive terminal events:
  1. Default (loss event)
  2. Prepayment (full payoff / refinance)

In standard mortgage performance, standard Kaplan-Meier curves for default
overestimate default risk because prepayment acts as a competing event that
removes borrowers from the risk pool. This module implements non-parametric
Aalen-Johansen / cause-specific hazard cumulative incidence estimation.

Run: PYTHONPATH=. python src/models/survival/competing_risks.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


def fit_competing_risks(panel_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit cause-specific hazard cumulative incidence models for default and prepayment.
    """
    log.info("Aggregating loan panel to event endpoints...")
    # Determine ultimate outcome per loan
    grouped = panel_df.groupby("loan_id").agg({
        "loan_age_months": "max",
        "default_flag": "max",
        "prepayment_flag": "max",
        "credit_score_band": "first",
    }).reset_index()

    # Define event status: 0 = Censored, 1 = Default, 2 = Prepayment
    def get_event_type(row):
        if row["default_flag"] == 1:
            return 1  # Default
        elif row["prepayment_flag"] == 1:
            return 2  # Prepayment
        return 0      # Active / Censored

    grouped["event_type"] = grouped.apply(get_event_type, axis=1)
    duration = grouped["loan_age_months"].values
    event = grouped["event_type"].values

    log.info(f"Cohort size: {len(grouped):,} loans. Defaults: {(event==1).sum():,}, Prepayments: {(event==2).sum():,}, Censored: {(event==0).sum():,}")

    # Fit cause-specific Nelson-Aalen estimators
    naf_default = NelsonAalenFitter()
    naf_default.fit(duration, event_observed=(event == 1), label="Default Hazard")

    naf_prepay = NelsonAalenFitter()
    naf_prepay.fit(duration, event_observed=(event == 2), label="Prepayment Hazard")

    # Overall survival function S(t) = exp(-(H_def(t) + H_prep(t)))
    times = np.sort(np.unique(duration))
    cum_haz_def = naf_default.cumulative_hazard_at_times(times).values.flatten()
    cum_haz_prep = naf_prepay.cumulative_hazard_at_times(times).values.flatten()

    total_haz = cum_haz_def + cum_haz_prep
    overall_survival = np.exp(-total_haz)

    # Compute discrete Cumulative Incidence Functions (CIF)
    # CIF_k(t) = sum_{u <= t} S(u-1) * dH_k(u)
    cif_default = []
    cif_prepayment = []

    running_cif_def = 0.0
    running_cif_prep = 0.0

    prev_haz_def = 0.0
    prev_haz_prep = 0.0

    for i, t in enumerate(times):
        s_prev = overall_survival[i - 1] if i > 0 else 1.0
        dh_def = max(0.0, cum_haz_def[i] - prev_haz_def)
        dh_prep = max(0.0, cum_haz_prep[i] - prev_haz_prep)

        running_cif_def += s_prev * dh_def
        running_cif_prep += s_prev * dh_prep

        cif_default.append(min(1.0, running_cif_def))
        cif_prepayment.append(min(1.0, running_cif_prep))

        prev_haz_def = cum_haz_def[i]
        prev_haz_prep = cum_haz_prep[i]

    # Sample milestones (Month 6, 12, 18, 24, 36)
    milestones = [6, 12, 18, 24, 36]
    cif_summary = []
    for m in milestones:
        idx = np.searchsorted(times, m, side="right") - 1
        idx = max(0, min(idx, len(times) - 1))
        cif_summary.append({
            "month": m,
            "cif_default": round(float(cif_default[idx]), 4),
            "cif_prepayment": round(float(cif_prepayment[idx]), 4),
            "survival_active": round(float(overall_survival[idx]), 4),
        })

    results = {
        "model_type": "Cause-Specific Cumulative Incidence Functions (CIF)",
        "cohort_loans": int(len(grouped)),
        "default_events": int((event == 1).sum()),
        "prepayment_events": int((event == 2).sum()),
        "censored_loans": int((event == 0).sum()),
        "cif_milestones": cif_summary,
    }

    out_path = MODELS_DIR / "competing_risks_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Competing risk modeling complete. Results saved → {out_path}")
    return results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error("Train dataset not found.")
        return

    df = pd.read_csv(train_path)
    res = fit_competing_risks(df)

    print("\n" + "=" * 60)
    print("COMPETING RISKS CUMULATIVE INCIDENCE MILESTONES")
    print("=" * 60)
    for m in res["cif_milestones"]:
        print(f"  Month {m['month']:02d}: CIF(Default)={m['cif_default']:.2%} | CIF(Prepayment)={m['cif_prepayment']:.2%} | Active={m['survival_active']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
