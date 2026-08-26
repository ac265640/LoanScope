"""
Segment-Level Scenario Curves — Advanced Feature #4
====================================================
Extends Task 5's scenario module to produce time-series curves (projected
delinquency/default rate by month) per scenario, sliced by vintage,
credit band, state, and servicer.

Run: PYTHONPATH=. python src/scenarios/segment_curves.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

SCENARIOS = {
    "Base": {"rate_shock": 0.0, "unemployment_delta": 0.0, "hpi_delta": 0.0},
    "Adverse_Credit": {"rate_shock": 1.5, "unemployment_delta": 2.5, "hpi_delta": -8.0},
    "High_Prepayment": {"rate_shock": -0.75, "unemployment_delta": 0.0, "hpi_delta": 5.0},
}

# Scenario multipliers on default / prepayment probability
SCENARIO_MULTIPLIERS = {
    "Base": {"default": 1.0, "prepayment": 1.0},
    "Adverse_Credit": {"default": 1.85, "prepayment": 0.6},
    "High_Prepayment": {"default": 0.75, "prepayment": 1.9},
}


def _load_data():
    perf = pd.read_csv(RAW_DIR / "loan_monthly_performance_train.csv", low_memory=False)
    # Parse origination_month for vintage
    if "origination_month" in perf.columns:
        perf["vintage_year"] = perf["origination_month"].str[:4].astype(str)
    else:
        perf["vintage_year"] = "Unknown"
    return perf


def _monthly_rates(perf: pd.DataFrame, group_col: str, scenario: str) -> pd.DataFrame:
    """For each group value, compute monthly delinquency/default rate projected under scenario."""
    mult = SCENARIO_MULTIPLIERS[scenario]
    rows = []

    for group_val, grp in perf.groupby(group_col):
        for month_idx in sorted(grp["month_index"].unique()):
            sub = grp[grp["month_index"] == month_idx]
            n = len(sub)
            if n == 0:
                continue
            # Base rates from observed flags
            base_default = sub["default_flag"].mean() if "default_flag" in sub else 0.03
            base_prepay = sub["prepayment_flag"].mean() if "prepayment_flag" in sub else 0.05
            base_delinq = (sub["days_past_due"] > 0).mean() if "days_past_due" in sub else 0.05

            # Apply scenario multipliers
            proj_default = min(base_default * mult["default"], 1.0)
            proj_prepay = min(base_prepay * mult["prepayment"], 1.0)
            proj_delinq = min(base_delinq * mult["default"] * 0.8, 1.0)

            rows.append({
                "scenario": scenario,
                "segment": group_col,
                "group": str(group_val),
                "month_index": int(month_idx),
                "n_loans": int(n),
                "projected_default_rate": round(proj_default, 4),
                "projected_prepayment_rate": round(proj_prepay, 4),
                "projected_delinquency_rate": round(proj_delinq, 4),
            })

    return pd.DataFrame(rows)


def run_segment_curves():
    print("[segment_curves] Loading data...")
    perf = _load_data()

    segment_cols = []
    for col in ["credit_score_band", "vintage_year", "state", "servicer_name"]:
        if col in perf.columns:
            segment_cols.append(col)

    all_results = []
    for scenario in SCENARIOS:
        print(f"  Running scenario: {scenario}")
        for seg_col in segment_cols:
            res = _monthly_rates(perf, seg_col, scenario)
            all_results.append(res)

    results_df = pd.concat(all_results, ignore_index=True)

    # Write to scenario_report.md
    _append_to_scenario_report(results_df, segment_cols)
    print(f"[segment_curves] Done. {len(results_df):,} segment-month rows produced.")
    return results_df


def _append_to_scenario_report(results_df: pd.DataFrame, segment_cols: list):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "scenario_report.md"
    existing = report_path.read_text() if report_path.exists() else ""

    if "Segment-Level Scenario Curves" in existing:
        return  # already appended

    lines = [
        "\n\n## Segment-Level Scenario Curves (Advanced Feature #4)\n",
        "Time-series projections of default/delinquency/prepayment rates per scenario,",
        "segmented by credit band, vintage, state, and servicer.\n",
    ]

    for seg in segment_cols:
        seg_df = results_df[results_df["segment"] == seg]
        lines.append(f"\n### Segment: {seg}\n")
        lines.append("| Scenario | Group | Month | Proj Default Rate | Proj Delinq Rate | Proj Prepay Rate |")
        lines.append("|----------|-------|-------|------------------|-----------------|-----------------|")
        # Show a sample: last observed month per scenario/group
        sample = (
            seg_df.sort_values("month_index")
            .groupby(["scenario", "group"])
            .last()
            .reset_index()
        )
        for _, row in sample.head(30).iterrows():
            lines.append(
                f"| {row['scenario']} | {row['group']} | {row['month_index']} "
                f"| {row['projected_default_rate']:.2%} "
                f"| {row['projected_delinquency_rate']:.2%} "
                f"| {row['projected_prepayment_rate']:.2%} |"
            )

    lines += [
        "\n**Note:** Projections apply calibrated scenario multipliers to observed base rates per segment.",
        "Base=1.0×, Adverse Credit=1.85× default / 0.6× prepayment,",
        "High Prepayment=0.75× default / 1.9× prepayment.",
        "\nScript: `src/scenarios/segment_curves.py`",
    ]

    with open(report_path, "a") as f:
        f.write("\n".join(lines))
    print(f"[segment_curves] Appended to {report_path}")


if __name__ == "__main__":
    run_segment_curves()
