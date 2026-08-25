"""
Data Intelligence & Profiling Runner
====================================
Orchestrates Task 1 profiling suite and outputs `reports/data_intelligence_report.md`.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

from src.profiling.profiler import profile_dataframe
from src.profiling.missingness import analyze_missingness
from src.profiling.outlier_detector import run_comprehensive_outlier_audit
from src.profiling.date_validator import validate_date_relationships
from src.profiling.correlation import compute_correlations
from src.profiling.relationship_breaks import detect_relationship_breaks
from src.profiling.drift import compute_dataset_drift
from src.profiling.quality_scorer import evaluate_batch_quality
from src.profiling.source_conflict import detect_source_conflicts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_markdown_report(
    profile_train: dict,
    missing_results: dict,
    outlier_results: dict,
    date_results: dict,
    corr_results: dict,
    break_results: dict,
    drift_results: dict,
    dq_results: dict,
    conflict_results: dict,
) -> str:
    """Generate comprehensive markdown documentation for data intelligence."""
    md = []
    md.append("# Data Intelligence & Profiling Report")
    md.append("\n**Project**: Intain Campus FinTech Challenge 2026 — AI Track")
    md.append("**System**: Loan Performance Intelligence Engine")
    md.append(f"**Generated On**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    md.append("\n---\n")

    # 1. Executive Summary
    md.append("## 1. Executive Summary & Batch Data Quality Score\n")
    md.append(f"- **Total Train Panel Records**: `{profile_train['n_rows']:,}`")
    md.append(f"- **Total Inspected Columns**: `{profile_train['n_columns']}`")
    md.append(f"- **Portfolio Quality Grade**: **{dq_results['batch_quality_grade']}**")
    md.append(f"- **Mean Data Quality Score**: `{dq_results['batch_mean_dq_score']:.2f} / 100.0` (Median: `{dq_results['batch_median_dq_score']:.2f}`)")
    md.append(f"- **Pristine Quality Records (>=90 score)**: `{dq_results['score_distribution']['pristine_gte_90_pct']}%`")
    md.append(f"- **Degraded Records (<75 score)**: `{dq_results['score_distribution']['degraded_lt_75_pct']}%`")
    md.append("\n> [!IMPORTANT]\n> Data quality evaluation uses a strictly documented penalty framework covering temporal validity (-35), balance/status contradictions (-30), default/DPD mismatches (-25), and feature missingness (-10 to -5).\n")

    # 2. Missing-Value Mechanism Analysis
    md.append("## 2. Missing-Value Pattern & Mechanism Analysis (MCAR / MAR / MNAR)\n")
    md.append("| Column Name | Missing Count | Missing % | Inferred Mechanism | Statistical Diagnosis & Justification |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for col, data in missing_results["missing_summary"].items():
        mech_info = missing_results["mechanism_classification"].get(col, {})
        mech = mech_info.get("inferred_mechanism", "MCAR")
        rationale = mech_info.get("rationale", "Standard distribution.")
        md.append(f"| `{col}` | {data['missing_count']:,} | {data['missing_pct']}% | **{mech}** | {rationale} |")

    if missing_results.get("vintage_missingness"):
        md.append("\n### Missingness Breakdown by Origination Vintage Cohort")
        md.append("Historical analysis confirms non-uniform missingness in underwriting data across vintages:")
        for col, v_data in missing_results["vintage_missingness"].items():
            md.append(f"\n- **`{col}` Missing Rate by Vintage Year**:")
            for yr, rate in sorted(v_data.items()):
                md.append(f"  - `{yr}`: `{rate}%`")

    # 3. Outlier Detection
    md.append("\n## 3. Univariate & Multivariate Outlier Detection\n")
    md.append("### Univariate Outlier Summary (Tukey's IQR & Z-Score Analysis)\n")
    md.append("| Feature | IQR Lower Bound | IQR Upper Bound | IQR Outlier Count (%) | Z-Score (>3σ) Outliers (%) | Extreme Value Detected |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for col, iqr_data in outlier_results["univariate_iqr"].items():
        z_data = outlier_results["univariate_zscore"].get(col, {})
        max_ext = iqr_data.get("extreme_max_outlier")
        ext_str = f"`{max_ext}`" if max_ext is not None else "None"
        md.append(f"| `{col}` | `{iqr_data.get('lower_bound')}` | `{iqr_data.get('upper_bound')}` | {iqr_data.get('n_outliers', 0):,} ({iqr_data.get('pct_outliers', 0.0)}%) | {z_data.get('n_outliers', 0):,} ({z_data.get('pct_outliers', 0.0)}%) | {ext_str} |")

    iso_res = outlier_results.get("multivariate_isolation_forest", {})
    md.append("\n### Multivariate Outlier Analysis (Isolation Forest)")
    md.append(f"- **Algorithm**: Isolation Forest (100 estimators, contamination={iso_res.get('contamination', 0.01)})")
    md.append(f"- **Evaluated Features**: `{', '.join(iso_res.get('features_evaluated', []))}`")
    md.append(f"- **Multivariate Anomalies Flagged**: `{iso_res.get('n_anomalies_flagged', 0):,}` (`{iso_res.get('pct_anomalies', 0.0)}%` of panel)")
    md.append(f"- **Anomaly Score Distribution**: Min=`{iso_res.get('score_min')}`, Median=`{iso_res.get('score_median')}`, Max=`{iso_res.get('score_max')}`")

    # 4. Temporal Integrity & Date Relationships
    md.append("\n## 4. Temporal Integrity & Date Relationship Audit\n")
    md.append(f"- **Total Rows Audited**: `{date_results['total_rows_inspected']:,}`")
    md.append(f"- **Chronologically Anomalous Rows**: `{date_results['anomalous_rows_count']:,}` (`{date_results['anomalous_rows_pct']}%`)")
    for code, chk in date_results["checks"].items():
        md.append(f"\n#### Check: {code} (`{chk['severity']}` Severity)")
        md.append(f"- **Description**: {chk['description']}")
        md.append(f"- **Violations**: `{chk['violation_count']:,}` (`{chk['violation_pct']}%`)")
        if chk.get("examples"):
            md.append("- **Sample Flagged Records**:")
            for ex in chk["examples"]:
                md.append(f"  - Loan `{ex['loan_id']}`: Originated `{ex['origination_month']}`, Reported `{ex['reporting_month']}`")

    # 5. Cross-Column Invariant Violations
    md.append("\n## 5. Cross-Column Relationship-Break Detection\n")
    md.append("| Violation Code | Description | Severity | Violation Count | Violation % |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for code, brk in break_results["violation_types"].items():
        md.append(f"| `{code}` | {brk['name']} | `{brk['severity']}` | {brk['count']:,} | {brk['pct']}% |")

    # 6. Correlations & Associations
    md.append("\n## 6. Correlation & Multicollinearity Analysis\n")
    md.append("### High Pearson Correlation Pairs (|r| >= 0.40)")
    if corr_results["numeric_pearson_high_pairs"]:
        md.append("| Feature 1 | Feature 2 | Pearson r | Strength |")
        md.append("| :--- | :--- | :--- | :--- |")
        for pair in corr_results["numeric_pearson_high_pairs"]:
            md.append(f"| `{pair['feature_1']}` | `{pair['feature_2']}` | `{pair['pearson_r']}` | {pair['strength']} |")
    else:
        md.append("*No extreme continuous collinearities exceeding threshold.*")

    md.append("\n### Categorical Association (Cramér's V >= 0.15)")
    if corr_results["categorical_cramers_v_high_pairs"]:
        md.append("| Feature 1 | Feature 2 | Cramér's V | Association Strength |")
        md.append("| :--- | :--- | :--- | :--- |")
        for pair in corr_results["categorical_cramers_v_high_pairs"]:
            md.append(f"| `{pair['feature_1']}` | `{pair['feature_2']}` | `{pair['cramers_v']}` | {pair['association_strength']} |")
    else:
        md.append("*No high categorical associations exceeding threshold.*")

    # 7. Train vs. Test Distributional Drift
    md.append("\n## 7. Train vs. Test Distributional Drift (PSI & KS-Test)\n")
    md.append(f"- **Train Cohort**: `{drift_results['n_train_records']:,}` records (Vintages <= 2021-12)")
    md.append(f"- **Test Cohort**: `{drift_results['n_test_records']:,}` records (Vintages >= 2022-01)")
    md.append(f"- **Stable Features (PSI < 0.10)**: `{len(drift_results['stable_features'])}` features ({', '.join([f'`{f}`' for f in drift_results['stable_features'][:6]])}...)")
    md.append(f"- **Moderate Drift Features (0.10 <= PSI < 0.25)**: `{len(drift_results['moderate_drift_features'])}` ({', '.join([f'`{f}`' for f in drift_results['moderate_drift_features']]) if drift_results['moderate_drift_features'] else 'None'})")
    md.append(f"- **High Drift Features (PSI >= 0.25)**: `{len(drift_results['high_drift_features'])}` ({', '.join([f'`{f}`' for f in drift_results['high_drift_features']]) if drift_results['high_drift_features'] else 'None'})")

    md.append("\n| Feature | Type | Metric 1 (PSI / Max Shift) | Metric 2 (KS Stat / p-val) | Drift Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for f_name, met in drift_results["feature_metrics"].items():
        if met["type"] == "numeric":
            md.append(f"| `{f_name}` | Numeric | PSI: `{met['psi']}` | KS: `{met['ks_statistic']}` (p={met['ks_pvalue']:.2e}) | **`{met['status']}`** |")
        else:
            md.append(f"| `{f_name}` | Categorical | Max Shift: `{met['max_category_shift_pct']}%` | N/A | **`{met['status']}`** |")

    # 8. Source Conflict Analysis
    md.append("\n## 8. Source Conflict & Multi-Feed Reconciliation\n")
    md.append(f"- **Servicer Update Records Ingested**: `{conflict_results.get('n_servicer_records', 0):,}`")
    md.append(f"- **Matched Primary Records**: `{conflict_results.get('n_matched_records', 0):,}`")
    md.append(f"- **Total Conflicting Records**: `{conflict_results.get('total_conflicting_records', 0):,}` (`{conflict_results.get('conflict_rate_pct', 0.0)}%`)")
    md.append(f"- **Unpaid Principal Balance Discrepancies**: `{conflict_results.get('balance_discrepancies_count', 0):,}` (`{conflict_results.get('balance_discrepancies_pct', 0.0)}%`)")
    md.append(f"- **Performance Status Contradictions**: `{conflict_results.get('status_discrepancies_count', 0):,}` (`{conflict_results.get('status_discrepancies_pct', 0.0)}%`)")

    if conflict_results.get("sample_conflicts"):
        md.append("\n### Sample Discrepancy Records Between Feeds:")
        md.append("| Loan ID | Reporting Month | Primary Balance | Servicer Balance | Diff ($) | Primary Status | Servicer Status |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for sc in conflict_results["sample_conflicts"][:5]:
            md.append(f"| `{sc['loan_id']}` | `{sc['reporting_month']}` | `${sc['primary_balance']:,.2f}` | `${sc['servicer_balance']:,.2f}` | `${sc['balance_diff']:,.2f}` | `{sc['primary_status']}` | `{sc['servicer_status']}` |")

    # 9. Remediation Strategy
    md.append("\n## 9. Data Remediation & Preprocessing Architecture\n")
    md.append("1. **Time-Aware Cleaning**: Exclude records with critical date anomalies (`reporting_month < origination_month`) from training features.")
    md.append("2. **Missing Value Imputation**: Target encode categorical missingness with explicit `'<MISSING>'` tokens for MNAR `credit_score_band`; median imputation with missingness indicator for `interest_rate`.")
    md.append("3. **Cross-Column Consistency Logic**: Standardize status priority rules; enforce balance truncation for terminal states (`Paid Off` -> `$0`).")
    md.append("4. **Robust Scaling & Winsorization**: Winsorize continuous variables (`interest_rate` at 99.5th percentile, `days_past_due` at 180) to prevent outlier degradation in linear/distance baselines.")
    md.append("5. **Drift-Adaptive Feature Selection**: Retain robust vintage-agnostic relative ratios (e.g. `current_balance / original_balance`, `rate_to_market_spread`) over raw nominal levels.")

    return "\n".join(md)


def main():
    log.info("Starting Data Intelligence & Profiling Runner...")
    train_path = RAW_DIR / "loan_monthly_performance_train.csv"
    test_path = RAW_DIR / "loan_monthly_performance_test.csv"
    servicer_path = RAW_DIR / "servicer_updates.csv"

    if not train_path.exists():
        log.error(f"Train data file not found at {train_path}. Please run data generation first.")
        sys.exit(1)

    log.info("Loading datasets for profiling...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    servicer_df = pd.read_csv(servicer_path) if servicer_path.exists() else pd.DataFrame()

    log.info(f"Profiling Train ({len(train_df):,} rows) and Test ({len(test_df):,} rows)...")
    profile_train = profile_dataframe(train_df)
    missing_results = analyze_missingness(train_df)
    outlier_results = run_comprehensive_outlier_audit(train_df)
    date_results = validate_date_relationships(train_df)
    corr_results = compute_correlations(train_df)
    break_results = detect_relationship_breaks(train_df)
    drift_results = compute_dataset_drift(train_df, test_df)
    dq_results = evaluate_batch_quality(train_df)
    conflict_results = detect_source_conflicts(train_df, servicer_df) if not servicer_df.empty else {}

    log.info("Compiling markdown report...")
    md_report = generate_markdown_report(
        profile_train,
        missing_results,
        outlier_results,
        date_results,
        corr_results,
        break_results,
        drift_results,
        dq_results,
        conflict_results,
    )

    out_file = REPORTS_DIR / "data_intelligence_report.md"
    with open(out_file, "w") as f:
        f.write(md_report)

    log.info(f"✅ Data Intelligence Report successfully generated at {out_file}")


if __name__ == "__main__":
    main()
