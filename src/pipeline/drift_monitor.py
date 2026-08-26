"""
Feature Drift Monitoring Dashboard
====================================
Monitors population stability and distributional drift between the training baseline
and incoming scoring batches. Computes PSI (Population Stability Index) and KS statistic
per feature, flags unstable features, and writes a structured drift report.

This module supports ongoing model governance after initial deployment.

Run: PYTHONPATH=. python src/pipeline/drift_monitor.py
"""

import json
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"


# PSI thresholds (industry standard)
PSI_STABLE = 0.10        # < 0.10 = no significant change
PSI_MODERATE_DRIFT = 0.25  # 0.10 – 0.25 = moderate shift, monitor
PSI_SEVERE_DRIFT = 0.25   # > 0.25 = significant shift, model revalidation needed


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between expected (train) and actual (test) distributions.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    # Create decile-based buckets from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid log(0) with small floor
    expected_pcts = np.clip(expected_pcts, 1e-6, None)
    actual_pcts = np.clip(actual_pcts, 1e-6, None)

    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi)


def compute_ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Kolmogorov-Smirnov test statistic between two distributions."""
    ks_stat, _ = stats.ks_2samp(expected, actual)
    return float(ks_stat)


def classify_drift(psi: float) -> str:
    if psi < PSI_STABLE:
        return "Stable"
    elif psi < PSI_MODERATE_DRIFT:
        return "Moderate Drift"
    else:
        return "Severe Drift"


def run_drift_monitoring(train_features: pd.DataFrame, test_features: pd.DataFrame) -> dict:
    """
    Run full drift monitoring across all engineered features.
    Returns structured drift report with PSI + KS per feature.
    """
    feature_cols = get_feature_columns()
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(train_features[f])]

    log.info(f"Running drift monitoring on {len(numeric_features)} numeric features...")

    drift_results = {}
    flagged_features = []

    for feat in numeric_features:
        train_vals = train_features[feat].dropna().values
        test_vals = test_features[feat].dropna().values

        if len(train_vals) < 50 or len(test_vals) < 50:
            continue

        psi = compute_psi(train_vals, test_vals)
        ks = compute_ks(train_vals, test_vals)
        drift_label = classify_drift(psi)

        drift_results[feat] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks, 4),
            "drift_status": drift_label,
            "train_mean": round(float(train_vals.mean()), 4),
            "test_mean": round(float(test_vals.mean()), 4),
            "train_std": round(float(train_vals.std()), 4),
            "test_std": round(float(test_vals.std()), 4),
            "mean_shift_pct": round(
                abs(test_vals.mean() - train_vals.mean()) / (abs(train_vals.mean()) + 1e-9) * 100, 2
            ),
        }

        if drift_label != "Stable":
            flagged_features.append(feat)

    # Rank by PSI descending
    sorted_results = dict(sorted(drift_results.items(), key=lambda x: x[1]["psi"], reverse=True))

    severe = [f for f, v in drift_results.items() if v["drift_status"] == "Severe Drift"]
    moderate = [f for f, v in drift_results.items() if v["drift_status"] == "Moderate Drift"]

    summary = {
        "total_features_monitored": len(numeric_features),
        "stable_features": len(numeric_features) - len(flagged_features),
        "moderate_drift_features": len(moderate),
        "severe_drift_features": len(severe),
        "severe_drift_list": severe,
        "moderate_drift_list": moderate,
        "overall_portfolio_stability": "PASS" if len(severe) == 0 else "WARN" if len(severe) < 3 else "FAIL",
        "per_feature_drift": sorted_results,
    }

    return summary


def write_drift_report(drift_summary: dict):
    """Write human-readable drift report to reports/drift_monitoring_report.md."""
    report_path = REPORTS_DIR / "drift_monitoring_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    severe = drift_summary["severe_drift_list"]
    moderate = drift_summary["moderate_drift_list"]
    stability = drift_summary["overall_portfolio_stability"]

    lines = [
        "# Feature Drift Monitoring Report",
        "",
        "## Portfolio Stability Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Overall Status | **{stability}** |",
        f"| Features Monitored | {drift_summary['total_features_monitored']} |",
        f"| Stable Features | {drift_summary['stable_features']} |",
        f"| Moderate Drift | {drift_summary['moderate_drift_features']} |",
        f"| Severe Drift | {drift_summary['severe_drift_features']} |",
        "",
        "> PSI Thresholds: Stable < 0.10 | Moderate: 0.10–0.25 | Severe: > 0.25",
        "",
    ]

    if severe:
        lines += [
            "## ⚠️ Severe Drift Features (PSI > 0.25) — Model Revalidation Recommended",
            "",
        ]
        for feat in severe:
            r = drift_summary["per_feature_drift"][feat]
            lines.append(
                f"- **{feat}**: PSI={r['psi']:.4f}, KS={r['ks_statistic']:.4f}, "
                f"Mean shift={r['mean_shift_pct']:.1f}% "
                f"(Train μ={r['train_mean']:.4f} → Test μ={r['test_mean']:.4f})"
            )
        lines.append("")

    if moderate:
        lines += [
            "## ⚡ Moderate Drift Features (PSI 0.10–0.25) — Monitor Closely",
            "",
        ]
        for feat in moderate:
            r = drift_summary["per_feature_drift"][feat]
            lines.append(
                f"- **{feat}**: PSI={r['psi']:.4f}, KS={r['ks_statistic']:.4f}, "
                f"Mean shift={r['mean_shift_pct']:.1f}%"
            )
        lines.append("")

    lines += [
        "## Top 15 Features by PSI",
        "",
        "| Feature | PSI | KS Stat | Status | Mean Shift % |",
        "|---|---|---|---|---|",
    ]
    for feat, r in list(drift_summary["per_feature_drift"].items())[:15]:
        lines.append(
            f"| {feat} | {r['psi']:.4f} | {r['ks_statistic']:.4f} | "
            f"{r['drift_status']} | {r['mean_shift_pct']:.1f}% |"
        )

    report_path.write_text("\n".join(lines))
    log.info(f"✅ Drift monitoring report saved → {report_path}")


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    test_path = RAW_DIR / "loan_monthly_performance_test.csv"

    if not train_path.exists() or not test_path.exists():
        log.error("Data files not found. Run `make data` first.")
        return

    log.info("Loading datasets for drift monitoring...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # Use the train cohort as the baseline reference distribution
    train_df, _, _ = time_aware_cohort_split(train_raw, val_cutoff="2020-01-01", test_cutoff="2099-01-01")
    train_features = engineer_panel_features(train_df)
    test_features = engineer_panel_features(test_raw)

    drift_summary = run_drift_monitoring(train_features, test_features)

    # Save JSON summary
    out_json = MODELS_DIR / "drift_monitoring_results.json"
    with open(out_json, "w") as f:
        json.dump(drift_summary, f, indent=2)
    log.info(f"✅ Drift results saved → {out_json}")

    write_drift_report(drift_summary)

    # Print dashboard summary
    print("\n" + "=" * 60)
    print("FEATURE DRIFT MONITORING DASHBOARD")
    print("=" * 60)
    print(f"  Overall Status    : {drift_summary['overall_portfolio_stability']}")
    print(f"  Features Monitored: {drift_summary['total_features_monitored']}")
    print(f"  Stable            : {drift_summary['stable_features']}")
    print(f"  Moderate Drift    : {drift_summary['moderate_drift_features']}")
    print(f"  Severe Drift      : {drift_summary['severe_drift_features']}")
    if drift_summary["severe_drift_list"]:
        print(f"  SEVERE FEATURES   : {', '.join(drift_summary['severe_drift_list'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
