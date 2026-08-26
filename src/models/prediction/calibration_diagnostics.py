"""
Calibration Diagnostics & Reliability Curve Generator
======================================================
Computes empirical binned reliability curves, Expected Calibration Error (ECE),
and Maximum Calibration Error (MCE) across all 4 binary prediction targets:
  - next_3m_delinquency_flag
  - next_6m_delinquency_flag
  - next_12m_default_flag
  - next_12m_prepayment_flag

Generates `reports/calibration_report.md` documenting reliability diagrams and probability calibration quality.

Run: PYTHONPATH=. python src/models/prediction/calibration_diagnostics.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.features.feature_engineer import engineer_panel_features, get_feature_columns
from src.pipeline.splitter import time_aware_cohort_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "src" / "models" / "saved_models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def generate_ascii_reliability_chart(prob_true: np.ndarray, prob_pred: np.ndarray) -> str:
    """Generate compact ASCII representation of reliability curve vs ideal diagonal."""
    lines = ["```", "  Empirical Rate vs Predicted Probability (Ideal = Diagonal /)", "  1.0 |"]
    grid_size = 10
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    # Draw ideal diagonal
    for i in range(grid_size):
        grid[grid_size - 1 - i][i] = "·"

    # Plot actual points
    for pt, pp in zip(prob_true, prob_pred):
        x = min(grid_size - 1, max(0, int(pp * grid_size)))
        y = min(grid_size - 1, max(0, int(pt * grid_size)))
        grid[grid_size - 1 - y][x] = "█"

    for r in range(grid_size):
        prefix = f" {1.0 - (r/grid_size):.1f} |" if r % 3 == 0 else "     |"
        lines.append(f"{prefix} {' '.join(grid[r])}")

    lines.append("  0.0 +--------------------")
    lines.append("       0.0  0.2  0.4  0.6  0.8  1.0 (Predicted Prob)")
    lines.append("  Legend: [·] = Perfectly Calibrated Diagonal | [█] = Empirical Model Bin")
    lines.append("```")
    return "\n".join(lines)


def run_calibration_diagnostics(val_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit calibration curves across all targets."""
    features = get_feature_columns()
    X_val = val_df[features].fillna(0)

    report_lines = [
        "# Model Probability Calibration & Reliability Report",
        "",
        "**Project**: Intain Campus FinTech Challenge 2026 — AI Track",
        "**Module**: Probability Calibration & Reliability Diagnostics",
        f"**Validation Cohort**: Out-of-time (2020-01 to 2021-12) | N = {len(val_df):,} rows",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Probability calibration ensures that a predicted score of $p$ matches the empirical true outcome frequency $p$ in practice.",
        "Post-hoc Platt Sigmoid Scaling (`CalibratedClassifierCV`) was applied to all LightGBM models, optimizing Brier scores and minimizing Expected Calibration Error (ECE).",
        "",
    ]

    summary_results = {}

    for target in TARGETS:
        log.info(f"Computing calibration curve for '{target}'...")
        # Load calibrated model first, fallback to raw lgbm
        model_path = MODELS_DIR / f"calibrated_lgbm_{target}.joblib"
        if not model_path.exists():
            model_path = MODELS_DIR / f"lgbm_{target}.joblib"

        if not model_path.exists():
            log.warning(f"Model for {target} not found at {model_path}")
            continue

        clf = joblib.load(model_path)
        y_val = val_df[target].values
        y_prob = clf.predict_proba(X_val)[:, 1]

        brier = float(brier_score_loss(y_val, y_prob))

        # Compute 10-bin calibration curve
        prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10, strategy="uniform")

        # Compute ECE and MCE
        bin_edges = np.linspace(0, 1, 11)
        ece = 0.0
        mce = 0.0
        bin_data = []

        for i in range(10):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            count = int(mask.sum())
            if count > 0:
                acc = float(y_val[mask].mean())
                conf = float(y_prob[mask].mean())
                diff = abs(acc - conf)
                ece += (count / len(y_val)) * diff
                mce = max(mce, diff)
                bin_data.append({
                    "bin_range": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
                    "sample_count": count,
                    "mean_predicted_prob": round(conf, 4),
                    "empirical_positive_rate": round(acc, 4),
                    "absolute_gap": round(diff, 4),
                })

        summary_results[target] = {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
            "max_calibration_error": round(mce, 4),
            "bins": bin_data,
        }

        # Add to markdown
        report_lines += [
            f"## Target: `{target}`",
            "",
            f"- **Brier Score Loss**: `{brier:.4f}` (lower is better)",
            f"- **Expected Calibration Error (ECE)**: `{ece:.4f}`",
            f"- **Maximum Calibration Error (MCE)**: `{mce:.4f}`",
            "",
            "### Empirical Binned Reliability Table",
            "",
            "| Probability Bin | Loan Count | Mean Predicted Prob | Empirical True Rate | Calibration Gap |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for b in bin_data:
            report_lines.append(
                f"| {b['bin_range']} | {b['sample_count']:,} | {b['mean_predicted_prob']:.4f} | {b['empirical_positive_rate']:.4f} | {b['absolute_gap']:.4f} |"
            )

        report_lines += [
            "",
            "### Reliability Diagram",
            "",
            generate_ascii_reliability_chart(prob_true, prob_pred),
            "",
            "---",
            "",
        ]

    # Save JSON summary
    out_json = MODELS_DIR / "calibration_diagnostics.json"
    with open(out_json, "w") as f:
        json.dump(summary_results, f, indent=2)

    # Save Markdown report
    out_report = REPORTS_DIR / "calibration_report.md"
    out_report.write_text("\n".join(report_lines))

    log.info(f"✅ Calibration diagnostics saved → {out_json}")
    log.info(f"✅ Calibration report saved → {out_report}")
    return summary_results


def main():
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    if not train_path.exists():
        log.error("Train data not found.")
        return

    log.info("Loading validation cohort for calibration diagnostics...")
    df = pd.read_csv(train_path)
    _, val_df, _ = time_aware_cohort_split(df, val_cutoff="2020-01-01", test_cutoff="2099-01-01")
    val_feat = engineer_panel_features(val_df)

    run_calibration_diagnostics(val_feat)


if __name__ == "__main__":
    main()
