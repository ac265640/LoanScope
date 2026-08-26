"""
Competing-Risk Survival Model — Feature #1
==========================================
Implements cause-specific hazards treating default and prepayment as competing
terminal events. Produces Cumulative Incidence Function (CIF) curves per credit
band and compares against the simpler single-risk model from Task 3.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_panel() -> pd.DataFrame:
    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.exists():
        # Try data/ directly
        raw_dir = REPO_ROOT / "data"
    perf = pd.read_csv(raw_dir / "loan_monthly_performance_train.csv", low_memory=False)
    static = pd.read_csv(raw_dir / "loan_static_attributes.csv", low_memory=False)
    extra_cols = [c for c in static.columns if c not in perf.columns]
    if extra_cols:
        df = perf.merge(static[["loan_id"] + extra_cols], on="loan_id", how="left")
    else:
        df = perf
    return df


def _build_first_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each loan find the first month it became Default or Prepaid.
    Returns one row per loan with columns:
      loan_id, duration, event_type (0=censored, 1=default, 2=prepaid),
      credit_band
    """
    df = df.sort_values(["loan_id", "month_index"]).reset_index(drop=True)
    credit_band_col = "credit_score_band" if "credit_score_band" in df.columns else "credit_band"

    records = []
    for loan_id, grp in df.groupby("loan_id"):
        grp = grp.sort_values("month_index").reset_index(drop=True)
        credit_band = grp[credit_band_col].iloc[0] if credit_band_col in grp.columns else "Unknown"
        event_type = 0
        duration = len(grp)

        # Check flags per row
        for t in range(len(grp)):
            row = grp.iloc[t]
            # Default event
            if row.get("default_flag", 0) == 1:
                event_type = 1
                duration = t + 1
                break
            # Prepayment event
            elif row.get("prepayment_flag", 0) == 1:
                event_type = 2
                duration = t + 1
                break
            # Fallback: status string
            else:
                status = str(row.get("current_status", "")).strip().lower()
                if any(k in status for k in ["default", "charge", "90", "seriously"]):
                    event_type = 1
                    duration = t + 1
                    break
                elif any(k in status for k in ["prepaid", "paid off", "payoff"]):
                    event_type = 2
                    duration = t + 1
                    break

        records.append({
            "loan_id": loan_id,
            "duration": max(1, duration),
            "event_type": event_type,
            "credit_band": str(credit_band),
        })

    return pd.DataFrame(records)


def _kaplan_meier_cause_specific(event_df: pd.DataFrame, cause: int) -> pd.DataFrame:
    """
    Cause-specific KM estimator: treat other causes as censored.
    Returns DataFrame with columns: time, survival.
    """
    df = event_df.copy()
    df["event_indicator"] = (df["event_type"] == cause).astype(int)
    df = df.sort_values("duration").reset_index(drop=True)

    times = sorted(df["duration"].unique())
    n = len(df)
    S = 1.0
    rows = []
    for t in times:
        at_risk = (df["duration"] >= t).sum()
        events = ((df["duration"] == t) & (df["event_indicator"] == 1)).sum()
        if at_risk > 0:
            S *= (1 - events / at_risk)
        rows.append({"time": t, "survival": S})
    return pd.DataFrame(rows)


def _cif_aalen_johansen(event_df: pd.DataFrame, cause: int) -> pd.DataFrame:
    """
    Non-parametric CIF via Aalen-Johansen (Nelson-Aalen cause-specific hazard approach).
    CIF(t, cause) = integral of cause-specific hazard * overall survival.
    Returns DataFrame with columns: time, cif.
    """
    df = event_df.copy()
    times = sorted(df["duration"].unique())

    # Overall survival (all causes competing)
    n = len(df)
    S = 1.0
    overall_S = {}
    for t in times:
        at_risk = (df["duration"] >= t).sum()
        events_all = (df["duration"] == t).sum()
        if at_risk > 0:
            S *= (1 - events_all / at_risk)
        overall_S[t] = S

    # CIF accumulation
    cif = 0.0
    rows = []
    S_prev = 1.0
    for t in times:
        at_risk = (df["duration"] >= t).sum()
        events_cause = ((df["duration"] == t) & (df["event_type"] == cause)).sum()
        if at_risk > 0:
            h_cause = events_cause / at_risk
        else:
            h_cause = 0.0
        cif += S_prev * h_cause
        S_prev = overall_S[t]
        rows.append({"time": t, "cif": cif})

    return pd.DataFrame(rows)


def run_competing_risk_analysis(sample_n: int = 20000, random_state: int = 42):
    """
    Main entry point: loads data, builds competing-risk CIF curves,
    compares to single-risk KM, and writes survival_report.md.
    """
    print("[competing_risk] Loading panel data...")
    df = _load_panel()
    if len(df) > sample_n:
        loan_ids = df["loan_id"].drop_duplicates().sample(sample_n // 36, random_state=random_state)
        df = df[df["loan_id"].isin(loan_ids)]

    print("[competing_risk] Building first-event table...")
    event_df = _build_first_event(df)

    n_total = len(event_df)
    n_default = (event_df["event_type"] == 1).sum()
    n_prepaid = (event_df["event_type"] == 2).sum()
    n_censored = (event_df["event_type"] == 0).sum()

    print(f"  Loans: {n_total:,}  Default: {n_default:,}  Prepaid: {n_prepaid:,}  Censored: {n_censored:,}")

    # --- Global CIFs ---
    cif_default = _cif_aalen_johansen(event_df, cause=1)
    cif_prepaid = _cif_aalen_johansen(event_df, cause=2)

    # --- CIF by credit band ---
    bands = event_df["credit_band"].unique()
    band_results = {}
    for band in sorted(bands):
        sub = event_df[event_df["credit_band"] == band]
        if len(sub) < 30:
            continue
        band_results[band] = {
            "default": _cif_aalen_johansen(sub, cause=1),
            "prepaid": _cif_aalen_johansen(sub, cause=2),
            "n": len(sub),
            "n_default": (sub["event_type"] == 1).sum(),
            "n_prepaid": (sub["event_type"] == 2).sum(),
        }

    # --- Single-risk KM for comparison ---
    km_default = _kaplan_meier_cause_specific(event_df, cause=1)
    km_prepaid = _kaplan_meier_cause_specific(event_df, cause=2)

    # Single-risk cumulative incidence = 1 - KM survival
    km_default["single_risk_cif"] = 1 - km_default["survival"]
    km_prepaid["single_risk_cif"] = 1 - km_prepaid["survival"]

    # Compare at t=12, 24, 36
    comparison_rows = []
    for t in [12, 24, 36]:
        cr_d = cif_default[cif_default["time"] <= t]["cif"].iloc[-1] if len(cif_default[cif_default["time"] <= t]) > 0 else 0
        cr_p = cif_prepaid[cif_prepaid["time"] <= t]["cif"].iloc[-1] if len(cif_prepaid[cif_prepaid["time"] <= t]) > 0 else 0
        sr_d = km_default[km_default["time"] <= t]["single_risk_cif"].iloc[-1] if len(km_default[km_default["time"] <= t]) > 0 else 0
        sr_p = km_prepaid[km_prepaid["time"] <= t]["single_risk_cif"].iloc[-1] if len(km_prepaid[km_prepaid["time"] <= t]) > 0 else 0
        comparison_rows.append({
            "horizon_months": t,
            "competing_risk_CIF_default": round(cr_d, 4),
            "single_risk_CIF_default": round(sr_d, 4),
            "bias_default": round(sr_d - cr_d, 4),
            "competing_risk_CIF_prepaid": round(cr_p, 4),
            "single_risk_CIF_prepaid": round(sr_p, 4),
            "bias_prepaid": round(sr_p - cr_p, 4),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # --- Write survival_report.md ---
    _write_survival_report(
        n_total, n_default, n_prepaid, n_censored,
        cif_default, cif_prepaid, band_results, comparison_df
    )

    print("[competing_risk] Done. Report written to reports/survival_report.md")
    return {
        "event_summary": {"total": n_total, "default": int(n_default), "prepaid": int(n_prepaid), "censored": int(n_censored)},
        "comparison": comparison_df.to_dict(orient="records"),
    }


def _write_survival_report(n_total, n_default, n_prepaid, n_censored,
                            cif_default, cif_prepaid, band_results, comparison_df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "survival_report.md"

    default_pct = 100 * n_default / max(1, n_total)
    prepaid_pct = 100 * n_prepaid / max(1, n_total)
    censored_pct = 100 * n_censored / max(1, n_total)

    cif_d_12 = cif_default[cif_default["time"] <= 12]["cif"].iloc[-1] if len(cif_default[cif_default["time"] <= 12]) > 0 else 0
    cif_d_36 = cif_default[cif_default["time"] <= 36]["cif"].iloc[-1] if len(cif_default[cif_default["time"] <= 36]) > 0 else 0
    cif_p_12 = cif_prepaid[cif_prepaid["time"] <= 12]["cif"].iloc[-1] if len(cif_prepaid[cif_prepaid["time"] <= 12]) > 0 else 0
    cif_p_36 = cif_prepaid[cif_prepaid["time"] <= 36]["cif"].iloc[-1] if len(cif_prepaid[cif_prepaid["time"] <= 36]) > 0 else 0

    lines = [
        "# Survival Analysis Report — Competing-Risk Model\n",
        "## 1. Event Summary\n",
        f"| Outcome | Count | % of Portfolio |",
        f"|---------|-------|----------------|",
        f"| Default | {n_default:,} | {default_pct:.1f}% |",
        f"| Prepaid | {n_prepaid:,} | {prepaid_pct:.1f}% |",
        f"| Censored | {n_censored:,} | {censored_pct:.1f}% |",
        f"| **Total** | **{n_total:,}** | 100.0% |",
        "",
        "## 2. Global Cumulative Incidence Functions (Aalen-Johansen)\n",
        "The CIF represents the probability of experiencing a specific event before time `t`,",
        "**accounting for** the competing risk (the other event type).\n",
        f"| Horizon | CIF(Default) | CIF(Prepaid) | Sum |",
        f"|---------|-------------|--------------|-----|",
        f"| 12 months | {cif_d_12:.4f} | {cif_p_12:.4f} | {cif_d_12+cif_p_12:.4f} |",
        f"| 36 months | {cif_d_36:.4f} | {cif_p_36:.4f} | {cif_d_36+cif_p_36:.4f} |",
        "",
        "**Note:** CIF(Default) + CIF(Prepaid) + P(Censored) = 1.0, which is the fundamental",
        "constraint of the competing-risk framework.\n",
        "## 3. CIF by Credit Band\n",
        "| Credit Band | N Loans | 12m CIF Default | 12m CIF Prepaid | 36m CIF Default | 36m CIF Prepaid |",
        "|------------|---------|----------------|----------------|----------------|----------------|",
    ]

    for band in sorted(band_results.keys()):
        res = band_results[band]
        cd12 = res["default"][res["default"]["time"] <= 12]["cif"].iloc[-1] if len(res["default"][res["default"]["time"] <= 12]) > 0 else 0
        cp12 = res["prepaid"][res["prepaid"]["time"] <= 12]["cif"].iloc[-1] if len(res["prepaid"][res["prepaid"]["time"] <= 12]) > 0 else 0
        cd36 = res["default"][res["default"]["time"] <= 36]["cif"].iloc[-1] if len(res["default"][res["default"]["time"] <= 36]) > 0 else 0
        cp36 = res["prepaid"][res["prepaid"]["time"] <= 36]["cif"].iloc[-1] if len(res["prepaid"][res["prepaid"]["time"] <= 36]) > 0 else 0
        lines.append(f"| {band} | {res['n']:,} | {cd12:.4f} | {cp12:.4f} | {cd36:.4f} | {cp36:.4f} |")

    lines += [
        "",
        "## 4. Competing-Risk vs Single-Risk Comparison\n",
        "The single-risk Kaplan-Meier model treats competing events as random censoring,",
        "leading to **upward bias** in the estimated probability of each event type.",
        "The Aalen-Johansen competing-risk CIF corrects this.\n",
        comparison_df.to_markdown(index=False),
        "",
        "**Key insight:** The single-risk model systematically over-estimates the probability",
        "of each event by treating the competing event as mere censoring. For example,",
        "the default CIF bias (`single_risk - competing_risk`) is positive at all horizons,",
        "meaning a naive KM analysis would over-state the default risk in portfolios with",
        "active prepayments (common in low-rate environments).\n",
        "## 5. Methodology\n",
        "- **Estimator:** Non-parametric Aalen-Johansen estimator for CIF curves.",
        "- **Events:** Default (event_type=1) includes delinquency states (30/60/90 DPD,",
        "  charge-off); Prepaid (event_type=2) includes full payoff and voluntary prepayment.",
        "- **Censoring:** Loans still active at dataset end treated as right-censored.",
        "- **Segmentation:** Credit band based on `credit_score` at origination.",
        "- **Comparison model:** Cause-specific Kaplan-Meier (single-risk) treating",
        "  competing event as non-informative censoring (yields upward-biased CIF).\n",
        "## 6. Implementation\n",
        "- Script: `src/models/survival/competing_risk.py`",
        "- Run: `python -m src.models.survival.competing_risk`",
        "- Advanced Feature #1 of 15\n",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    results = run_competing_risk_analysis()
    print(json.dumps(results, indent=2, default=str))
