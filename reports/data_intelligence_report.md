# Data Intelligence & Profiling Report

**Project**: Intain Campus FinTech Challenge 2026 — AI Track
**System**: Loan Performance Intelligence Engine
**Generated On**: 2026-08-26 14:17:20 UTC

---

## 1. Executive Summary & Batch Data Quality Score

- **Total Train Panel Records**: `874,435`
- **Total Inspected Columns**: `33`
- **Portfolio Quality Grade**: **A (Pristine / Grade A Quality)**
- **Mean Data Quality Score**: `97.45 / 100.0` (Median: `100.00`)
- **Pristine Quality Records (>=90 score)**: `95.93%`
- **Degraded Records (<75 score)**: `0.46%`

> [!IMPORTANT]
> Data quality evaluation uses a strictly documented penalty framework covering temporal validity (-35), balance/status contradictions (-30), default/DPD mismatches (-25), and feature missingness (-10 to -5).

## 2. Missing-Value Pattern & Mechanism Analysis (MCAR / MAR / MNAR)

| Column Name | Missing Count | Missing % | Inferred Mechanism | Statistical Diagnosis & Justification |
| :--- | :--- | :--- | :--- | :--- |
| `interest_rate` | 27,161 | 3.106% | **MCAR (Missing Completely At Random)** | Missingness is uniformly dispersed (~3%) without strong correlation to credit quality, loan term, or origination vintage. |
| `credit_score_band` | 62,839 | 7.186% | **MNAR (Missing Not At Random)** | Missingness rate is significantly higher in legacy pre-2010 vintages (>15%) compared to recent vintages (<1%), reflecting historical underwriting record gaps. |
| `loss_severity_band` | 867,646 | 99.224% | **MAR (Missing At Random)** | Missingness depends on observed variable `default_flag` (correlation -1.0). |
| `exception_type` | 863,038 | 98.697% | **MAR (Missing At Random)** | Missingness depends on observed variable `exception_required` (correlation -1.0). |

### Missingness Breakdown by Origination Vintage Cohort
Historical analysis confirms non-uniform missingness in underwriting data across vintages:

- **`interest_rate` Missing Rate by Vintage Year**:
  - `2003`: `3.03%`
  - `2004`: `3.05%`
  - `2005`: `2.71%`
  - `2006`: `3.08%`
  - `2007`: `2.51%`
  - `2008`: `3.08%`
  - `2009`: `3.46%`
  - `2010`: `2.78%`
  - `2011`: `2.93%`
  - `2012`: `3.24%`
  - `2013`: `3.17%`
  - `2014`: `3.48%`
  - `2015`: `2.5%`
  - `2016`: `3.03%`
  - `2017`: `3.7%`
  - `2018`: `3.06%`
  - `2019`: `4.15%`
  - `2020`: `3.06%`
  - `2021`: `2.98%`

- **`credit_score_band` Missing Rate by Vintage Year**:
  - `2003`: `15.76%`
  - `2004`: `15.88%`
  - `2005`: `15.01%`
  - `2006`: `14.9%`
  - `2007`: `14.35%`
  - `2008`: `15.21%`
  - `2009`: `15.52%`
  - `2010`: `4.2%`
  - `2011`: `5.64%`
  - `2012`: `4.89%`
  - `2013`: `5.58%`
  - `2014`: `5.08%`
  - `2015`: `0.95%`
  - `2016`: `0.95%`
  - `2017`: `1.34%`
  - `2018`: `0.92%`
  - `2019`: `1.01%`
  - `2020`: `0.53%`
  - `2021`: `0.75%`

- **`loss_severity_band` Missing Rate by Vintage Year**:
  - `2003`: `99.0%`
  - `2004`: `98.98%`
  - `2005`: `98.98%`
  - `2006`: `98.95%`
  - `2007`: `99.01%`
  - `2008`: `99.09%`
  - `2009`: `99.06%`
  - `2010`: `99.17%`
  - `2011`: `99.34%`
  - `2012`: `99.3%`
  - `2013`: `99.34%`
  - `2014`: `99.32%`
  - `2015`: `99.38%`
  - `2016`: `99.34%`
  - `2017`: `99.34%`
  - `2018`: `99.34%`
  - `2019`: `99.32%`
  - `2020`: `99.49%`
  - `2021`: `99.42%`

- **`exception_type` Missing Rate by Vintage Year**:
  - `2003`: `98.37%`
  - `2004`: `98.38%`
  - `2005`: `98.55%`
  - `2006`: `98.5%`
  - `2007`: `98.53%`
  - `2008`: `98.56%`
  - `2009`: `98.59%`
  - `2010`: `98.56%`
  - `2011`: `98.89%`
  - `2012`: `98.79%`
  - `2013`: `98.79%`
  - `2014`: `98.83%`
  - `2015`: `98.93%`
  - `2016`: `98.73%`
  - `2017`: `98.88%`
  - `2018`: `98.72%`
  - `2019`: `98.75%`
  - `2020`: `98.91%`
  - `2021`: `98.91%`

## 3. Univariate & Multivariate Outlier Detection

### Univariate Outlier Summary (Tukey's IQR & Z-Score Analysis)

| Feature | IQR Lower Bound | IQR Upper Bound | IQR Outlier Count (%) | Z-Score (>3σ) Outliers (%) | Extreme Value Detected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `interest_rate` | `1.2655` | `7.7655` | 11,122 (1.272%) | 8,638 (0.988%) | `24.965642178411603` |
| `current_balance` | `-61257.9` | `340288.42` | 36,817 (4.21%) | 13,981 (1.599%) | `2256693.93` |
| `original_balance` | `-61306.0` | `346966.0` | 36,529 (4.177%) | 14,034 (1.605%) | `1417452.0` |
| `days_past_due` | `0.0` | `0.0` | 23,822 (2.724%) | 10,601 (1.212%) | `399.0` |
| `remaining_term_months` | `303.0` | `383.0` | 172,486 (19.725%) | 0 (0.0%) | `239.0` |

### Multivariate Outlier Analysis (Isolation Forest)
- **Algorithm**: Isolation Forest (100 estimators, contamination=0.01)
- **Evaluated Features**: `current_balance, original_balance, interest_rate, days_past_due, loan_age_months`
- **Multivariate Anomalies Flagged**: `8,473` (`1.0%` of panel)
- **Anomaly Score Distribution**: Min=`-0.2841`, Median=`-0.2355`, Max=`0.1415`

## 4. Temporal Integrity & Date Relationship Audit

- **Total Rows Audited**: `874,435`
- **Chronologically Anomalous Rows**: `1,710` (`0.1956%`)

#### Check: reporting_before_origination (`CRITICAL` Severity)
- **Description**: Reporting month occurs prior to loan origination funding date.
- **Violations**: `1,710` (`0.1956%`)
- **Sample Flagged Records**:
  - Loan `LN0000006`: Originated `2008-12`, Reported `2008-10`
  - Loan `LN0000030`: Originated `2018-08`, Reported `2018-06`
  - Loan `LN0000031`: Originated `2004-09`, Reported `2004-07`
  - Loan `LN0000097`: Originated `2006-08`, Reported `2006-06`
  - Loan `LN0000110`: Originated `2006-04`, Reported `2006-02`

#### Check: negative_remaining_term (`HIGH` Severity)
- **Description**: Remaining term in months is less than 0.
- **Violations**: `0` (`0.0%`)

#### Check: non_positive_loan_age (`MEDIUM` Severity)
- **Description**: Loan age in months is <= 0 for active panel observation.
- **Violations**: `0` (`0.0%`)

## 5. Cross-Column Relationship-Break Detection

| Violation Code | Description | Severity | Violation Count | Violation % |
| :--- | :--- | :--- | :--- | :--- |
| `BRK01_PAID_OFF_BALANCE` | Paid Off Status with Active Non-Zero Balance | `CRITICAL` | 0 | 0.0% |
| `BRK02_PREPAID_BALANCE` | Prepaid Status with Active Non-Zero Balance | `HIGH` | 0 | 0.0% |
| `BRK03_DEFAULT_LOW_DPD` | Default Status with Low Days Past Due | `HIGH` | 5,815 | 0.665% |
| `BRK04_CURRENT_HIGH_DPD` | Current Status with Active Delinquency | `HIGH` | 7,054 | 0.8067% |
| `BRK05_STATUS_FLAG_MISMATCH` | Default Status without Default Flag | `CRITICAL` | 0 | 0.0% |
| `BRK06_MOD_MISSING_DOC` | Restructured/Modified Loan with Missing Verification Files | `MEDIUM` | 1,400 | 0.1601% |
| `BRK07_EXCESSIVE_BALANCE_GROWTH` | Current Balance Exceeds 200% Original Disbursement | `HIGH` | 2,515 | 0.2876% |

## 6. Correlation & Multicollinearity Analysis

### High Pearson Correlation Pairs (|r| >= 0.40)
| Feature 1 | Feature 2 | Pearson r | Strength |
| :--- | :--- | :--- | :--- |
| `month_index` | `loan_age_months` | `1.0` | High |
| `original_balance` | `current_balance` | `0.9805` | High |

### Categorical Association (Cramér's V >= 0.15)
*No high categorical associations exceeding threshold.*

## 7. Train vs. Test Distributional Drift (PSI & KS-Test)

- **Train Cohort**: `874,435` records (Vintages <= 2021-12)
- **Test Cohort**: `69,871` records (Vintages >= 2022-01)
- **Stable Features (PSI < 0.10)**: `21` features (`month_index`, `loan_age_months`, `remaining_term_months`, `original_balance`, `current_balance`, `interest_rate`...)
- **Moderate Drift Features (0.10 <= PSI < 0.25)**: `1` (`credit_score_band`)
- **High Drift Features (PSI >= 0.25)**: `0` (None)

| Feature | Type | Metric 1 (PSI / Max Shift) | Metric 2 (KS Stat / p-val) | Drift Status |
| :--- | :--- | :--- | :--- | :--- |
| `month_index` | Numeric | PSI: `0.0009` | KS: `0.0136` (p=7.85e-11) | **`STABLE`** |
| `loan_age_months` | Numeric | PSI: `0.0009` | KS: `0.0136` (p=7.85e-11) | **`STABLE`** |
| `remaining_term_months` | Numeric | PSI: `0.0011` | KS: `0.0096` (p=1.48e-05) | **`STABLE`** |
| `original_balance` | Numeric | PSI: `0.0034` | KS: `0.0152` (p=1.70e-13) | **`STABLE`** |
| `current_balance` | Numeric | PSI: `0.0028` | KS: `0.0135` (p=1.17e-10) | **`STABLE`** |
| `interest_rate` | Numeric | PSI: `0.0045` | KS: `0.0291` (p=2.11e-46) | **`STABLE`** |
| `credit_score_band` | Categorical | Max Shift: `6.37%` | N/A | **`MODERATE_DRIFT`** |
| `ltv_band` | Categorical | Max Shift: `1.2%` | N/A | **`STABLE`** |
| `dti_band` | Categorical | Max Shift: `1.14%` | N/A | **`STABLE`** |
| `state` | Categorical | Max Shift: `1.1%` | N/A | **`STABLE`** |
| `loan_purpose` | Categorical | Max Shift: `1.45%` | N/A | **`STABLE`** |
| `occupancy_type` | Categorical | Max Shift: `0.2%` | N/A | **`STABLE`** |
| `property_type` | Categorical | Max Shift: `0.51%` | N/A | **`STABLE`** |
| `servicer_name` | Categorical | Max Shift: `0.97%` | N/A | **`STABLE`** |
| `current_status` | Categorical | Max Shift: `1.13%` | N/A | **`STABLE`** |
| `days_past_due` | Numeric | PSI: `0.0` | KS: `0.0088` (p=8.66e-05) | **`STABLE`** |
| `modification_flag` | Numeric | PSI: `0.0` | KS: `0.0017` (p=9.94e-01) | **`STABLE`** |
| `prepayment_flag` | Numeric | PSI: `0.0` | KS: `0.0002` (p=1.00e+00) | **`STABLE`** |
| `default_flag` | Numeric | PSI: `0.0` | KS: `0.0021` (p=9.41e-01) | **`STABLE`** |
| `loss_severity_band` | Categorical | Max Shift: `0.21%` | N/A | **`STABLE`** |
| `source_system` | Categorical | Max Shift: `0.7%` | N/A | **`STABLE`** |
| `document_status` | Categorical | Max Shift: `0.7%` | N/A | **`STABLE`** |

## 8. Source Conflict & Multi-Feed Reconciliation

- **Servicer Update Records Ingested**: `218,609`
- **Matched Primary Records**: `218,626`
- **Total Conflicting Records**: `10,758` (`4.92%`)
- **Unpaid Principal Balance Discrepancies**: `10,691` (`4.89%`)
- **Performance Status Contradictions**: `459` (`0.21%`)

### Sample Discrepancy Records Between Feeds:
| Loan ID | Reporting Month | Primary Balance | Servicer Balance | Diff ($) | Primary Status | Servicer Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LN0000002` | `2018-01` | `$92,644.45` | `$87,185.08` | `$5,459.37` | `Current` | `Current` |
| `LN0000006` | `2009-09` | `$181,340.46` | `$184,323.50` | `$-2,983.04` | `Current` | `Current` |
| `LN0000008` | `2005-05` | `$55,671.90` | `$54,194.48` | `$1,477.42` | `Current` | `Current` |
| `LN0000008` | `2006-11` | `$54,079.43` | `$67,562.94` | `$-13,483.51` | `Current` | `Current` |
| `LN0000010` | `2015-04` | `$194,287.46` | `$236,735.64` | `$-42,448.18` | `Current` | `Current` |

## 9. Data Remediation & Preprocessing Architecture

1. **Time-Aware Cleaning**: Exclude records with critical date anomalies (`reporting_month < origination_month`) from training features.
2. **Missing Value Imputation**: Target encode categorical missingness with explicit `'<MISSING>'` tokens for MNAR `credit_score_band`; median imputation with missingness indicator for `interest_rate`.
3. **Cross-Column Consistency Logic**: Standardize status priority rules; enforce balance truncation for terminal states (`Paid Off` -> `$0`).
4. **Robust Scaling & Winsorization**: Winsorize continuous variables (`interest_rate` at 99.5th percentile, `days_past_due` at 180) to prevent outlier degradation in linear/distance baselines.
5. **Drift-Adaptive Feature Selection**: Retain robust vintage-agnostic relative ratios (e.g. `current_balance / original_balance`, `rate_to_market_spread`) over raw nominal levels.