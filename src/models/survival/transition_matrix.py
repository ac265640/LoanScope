"""
Monthly Transition Matrix (Markov Chain) Model
==============================================
Estimates monthly multi-state transition probabilities across loan delinquency states:
Current, 30-59 DPD, 60-89 DPD, 90+ DPD, Default, Prepaid, Paid Off.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"

STATES_ORDER = ["Current", "30-59 DPD", "60-89 DPD", "90+ DPD", "Default", "Prepaid", "Paid Off"]


def compute_transition_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute empirical 1-month Markov state transition probability matrix P(S_{t+1} | S_t).
    """
    log.info("Computing empirical monthly transition probability matrix...")
    df = df.sort_values(["loan_id", "month_index"])

    df["current_state"] = df["current_status"]
    df["next_observed_state"] = df.groupby("loan_id")["current_status"].shift(-1)

    # Filter out terminal transitions where next state is NaN (end of panel)
    valid_transitions = df.dropna(subset=["current_state", "next_observed_state"])

    # Build transition counts
    crosstab_counts = pd.crosstab(
        valid_transitions["current_state"],
        valid_transitions["next_observed_state"],
    )

    # Reindex to ensure all standard states exist
    crosstab_counts = crosstab_counts.reindex(index=STATES_ORDER, columns=STATES_ORDER, fill_value=0)

    # Ensure terminal absorbing states (Default, Prepaid, Paid Off) stay at 1.0 on diagonal
    for terminal in ["Default", "Prepaid", "Paid Off"]:
        if crosstab_counts.loc[terminal].sum() == 0:
            crosstab_counts.loc[terminal, terminal] = 1000

    # Row normalize to get transition probabilities
    row_sums = crosstab_counts.sum(axis=1).replace(0, 1)
    trans_matrix = crosstab_counts.div(row_sums, axis=0)

    matrix_dict = {}
    for from_state in STATES_ORDER:
        matrix_dict[from_state] = {
            to_state: round(float(trans_matrix.loc[from_state, to_state]), 4)
            for to_state in STATES_ORDER
        }

    # Simulate multi-step matrix powers (e.g. 6-month and 12-month transition projections)
    P = trans_matrix.values
    P6 = np.linalg.matrix_power(P, 6)
    P12 = np.linalg.matrix_power(P, 12)

    p6_dict = {
        from_s: {to_s: round(float(P6[i, j]), 4) for j, to_s in enumerate(STATES_ORDER)}
        for i, from_s in enumerate(STATES_ORDER)
    }
    p12_dict = {
        from_s: {to_s: round(float(P12[i, j]), 4) for j, to_s in enumerate(STATES_ORDER)}
        for i, from_s in enumerate(STATES_ORDER)
    }

    results = {
        "one_month_transition_matrix": matrix_dict,
        "six_month_projected_matrix": p6_dict,
        "twelve_month_projected_matrix": p12_dict,
        "states": STATES_ORDER,
        "n_transitions_observed": len(valid_transitions),
    }

    out_file = MODELS_DIR / "transition_matrix_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"✅ Transition matrix computed and saved to {out_file}")
    return results


def main():
    raw_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not raw_path.exists():
        log.error(f"Train data not found at {raw_path}.")
        return

    df = pd.read_csv(raw_path)
    compute_transition_matrix(df)


if __name__ == "__main__":
    main()
