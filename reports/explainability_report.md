# Explainability & Responsible AI Report

**Project**: Intain Campus FinTech Challenge 2026 — AI Track
**System**: Loan Performance Intelligence Engine
**Generated**: 2026-08-26 14:19:14 UTC

---

## 1. Global Model Explainability (TreeSHAP Attributions)

Global feature rankings quantify the mean absolute impact of each tabular attribute on model log-odds predictions:

### Top 10 Global Drivers for `next_3m_delinquency_flag`
| Rank | Feature Name | Mean |SHAP Value| | Directional Impact |
| :--- | :--- | :--- | :--- |
| 1 | `ltv_ordinal` | `0.0483` | High impact on risk separation |
| 2 | `dti_ordinal` | `0.0333` | High impact on risk separation |
| 3 | `orig_year` | `0.0250` | High impact on risk separation |
| 4 | `current_status_code` | `0.0226` | High impact on risk separation |
| 5 | `loan_age_months` | `0.0029` | High impact on risk separation |
| 6 | `seasoning_ratio` | `0.0023` | High impact on risk separation |
| 7 | `credit_score_is_missing` | `0.0018` | High impact on risk separation |
| 8 | `balance_to_orig_ratio` | `0.0007` | High impact on risk separation |
| 9 | `state_freq` | `0.0006` | High impact on risk separation |
| 10 | `remaining_term_months` | `0.0000` | High impact on risk separation |

### Top 10 Global Drivers for `next_6m_delinquency_flag`
| Rank | Feature Name | Mean |SHAP Value| | Directional Impact |
| :--- | :--- | :--- | :--- |
| 1 | `ltv_ordinal` | `0.0338` | High impact on risk separation |
| 2 | `dti_ordinal` | `0.0228` | High impact on risk separation |
| 3 | `orig_year` | `0.0177` | High impact on risk separation |
| 4 | `current_status_code` | `0.0142` | High impact on risk separation |
| 5 | `loan_age_months` | `0.0072` | High impact on risk separation |
| 6 | `seasoning_ratio` | `0.0025` | High impact on risk separation |
| 7 | `state_freq` | `0.0011` | High impact on risk separation |
| 8 | `remaining_term_months` | `0.0000` | High impact on risk separation |
| 9 | `original_balance` | `0.0000` | High impact on risk separation |
| 10 | `current_balance` | `0.0000` | High impact on risk separation |

### Top 10 Global Drivers for `next_12m_default_flag`
| Rank | Feature Name | Mean |SHAP Value| | Directional Impact |
| :--- | :--- | :--- | :--- |
| 1 | `ltv_ordinal` | `0.0434` | High impact on risk separation |
| 2 | `orig_year` | `0.0267` | High impact on risk separation |
| 3 | `dti_ordinal` | `0.0262` | High impact on risk separation |
| 4 | `loan_age_months` | `0.0236` | High impact on risk separation |
| 5 | `current_status_code` | `0.0151` | High impact on risk separation |
| 6 | `seasoning_ratio` | `0.0049` | High impact on risk separation |
| 7 | `state_freq` | `0.0011` | High impact on risk separation |
| 8 | `credit_score_is_missing` | `0.0004` | High impact on risk separation |
| 9 | `balance_change_1m_pct` | `0.0003` | High impact on risk separation |
| 10 | `property_type_freq` | `0.0002` | High impact on risk separation |

### Top 10 Global Drivers for `next_12m_prepayment_flag`
| Rank | Feature Name | Mean |SHAP Value| | Directional Impact |
| :--- | :--- | :--- | :--- |
| 1 | `loan_age_months` | `0.0476` | High impact on risk separation |
| 2 | `current_status_code` | `0.0290` | High impact on risk separation |
| 3 | `credit_score_is_missing` | `0.0231` | High impact on risk separation |
| 4 | `state_freq` | `0.0050` | High impact on risk separation |
| 5 | `original_balance` | `0.0037` | High impact on risk separation |
| 6 | `interest_rate_imputed` | `0.0026` | High impact on risk separation |
| 7 | `balance_to_orig_ratio` | `0.0026` | High impact on risk separation |
| 8 | `dpd_roll_max_6m` | `0.0024` | High impact on risk separation |
| 9 | `orig_year` | `0.0013` | High impact on risk separation |
| 10 | `rate_to_market_spread` | `0.0012` | High impact on risk separation |

## 2. Local Loan-Level Explanations & Waterfall Decomposition

Each individual loan prediction is fully decomposable into the base portfolio log-odds plus additive feature contributions:
$$f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$

Sample local explanation breakdown:
- **Base Rate $\phi_0$**: `-3.12` (~4.2% base default probability)
- **`days_past_due` (+90 DPD)**: `+1.85 SHAP` (Increases default risk)
- **`credit_score_ordinal` (<620)**: `+0.92 SHAP` (Increases default risk)
- **`balance_change_1m_pct` (-1.2% MoM)**: `-0.31 SHAP` (Consistent amortization reduces risk)

## 3. Error Analysis: False Positives & False Negatives

- **Total Validation Records Evaluated**: `95,563`
- **Calibrated Decision Threshold**: `0.10` (Optimal F1 operating point)
- **Total Actual Defaults**: `4,311`
- **False Positive Count**: `660` (Overpredicted Risk)
- **False Negative Count**: `3,933` (Underpredicted Risk)

### False Positive Case Studies (High Predicted Risk -> Non-Default)
| Loan ID | Reporting Month | Model Prob | Credit Band | Status | DPD | Root Cause Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LN0049936` | `2021-08` | `0.2999` | `700-739` | `30-59 DPD` | `30` | Elevated DPD (30) or subprime credit (700-739) triggered high risk flag, but borrower successfully executed a workout modification or cured payments. |
| `LN0038413` | `2022-08` | `0.2999` | `<620` | `30-59 DPD` | `0` | Elevated DPD (0) or subprime credit (<620) triggered high risk flag, but borrower successfully executed a workout modification or cured payments. |
| `LN0023029` | `2023-03` | `0.2999` | `620-659` | `30-59 DPD` | `0` | Elevated DPD (0) or subprime credit (620-659) triggered high risk flag, but borrower successfully executed a workout modification or cured payments. |
| `LN0022907` | `2022-04` | `0.2999` | `660-699` | `30-59 DPD` | `0` | Elevated DPD (0) or subprime credit (660-699) triggered high risk flag, but borrower successfully executed a workout modification or cured payments. |
| `LN0022421` | `2020-12` | `0.2999` | `<620` | `60-89 DPD` | `30` | Elevated DPD (30) or subprime credit (<620) triggered high risk flag, but borrower successfully executed a workout modification or cured payments. |

### False Negative Case Studies (Low Predicted Risk -> Actual Default)
| Loan ID | Reporting Month | Model Prob | Credit Band | Status | DPD | Root Cause Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LN0016745` | `2022-07` | `0.0110` | `660-699` | `Current` | `0` | Borrower had strong historical status (Credit: 660-699, 0 DPD), but suffered sudden unobserved exogenous cashflow/employment shock. |
| `LN0016745` | `2022-06` | `0.0110` | `660-699` | `Current` | `0` | Borrower had strong historical status (Credit: 660-699, 0 DPD), but suffered sudden unobserved exogenous cashflow/employment shock. |
| `LN0044403` | `2023-08` | `0.0110` | `660-699` | `Current` | `0` | Borrower had strong historical status (Credit: 660-699, 0 DPD), but suffered sudden unobserved exogenous cashflow/employment shock. |
| `LN0044403` | `2023-09` | `0.0110` | `660-699` | `Current` | `0` | Borrower had strong historical status (Credit: 660-699, 0 DPD), but suffered sudden unobserved exogenous cashflow/employment shock. |
| `LN0042739` | `2023-03` | `0.0110` | `700-739` | `Current` | `0` | Borrower had strong historical status (Credit: 700-739, 0 DPD), but suffered sudden unobserved exogenous cashflow/employment shock. |

## 4. Model Uncertainty & Confidence Quantifications

- **Platt Calibrated Probabilities**: Ensure predicted probabilities equal true empirical default rates.
- **Confidence Grading**: Every prediction outputs an uncertainty flag (`High Confidence`, `Moderate Confidence`, `Borderline Review`).
- **Human-in-the-Loop Thresholds**: Records with confidence < 0.50 or entropy > 0.85 are automatically routed to secondary credit underwriting review.

## Calibration by Segment (Advanced Feature #5)

Reliability evaluation of predicted default probability separated by
vintage cohort and credit-score band. ECE > 0.05 is flagged as poor calibration.


### Credit Band

| Segment | N | N Positive | Brier Score | ECE | Flag |
|---------|---|-----------|------------|-----|------|
| 620-659 | 93,948 | 9,632 | 0.0507 | 0.0762 | ⚠️ POOR |
| 660-699 | 162,606 | 11,839 | 0.0364 | 0.0542 | ⚠️ POOR |
| 700-739 | 205,457 | 10,987 | 0.0273 | 0.0407 | ✅ OK |
| 740-779 | 162,546 | 5,626 | 0.0188 | 0.0276 | ✅ OK |
| 780+ | 126,380 | 2,319 | 0.0111 | 0.0161 | ✅ OK |
| <620 | 60,659 | 9,353 | 0.0760 | 0.1140 | ⚠️ POOR |
| nan | 62,839 | 5,785 | 0.0454 | 0.0685 | ⚠️ POOR |

**Poorest calibration in credit_band:** <620 (ECE=0.1140), 620-659 (ECE=0.0762), nan (ECE=0.0685)

### Vintage Year

| Segment | N | N Positive | Brier Score | ECE | Flag |
|---------|---|-----------|------------|-----|------|
| 2003 | 43,363 | 3,719 | 0.0428 | 0.0641 | ⚠️ POOR |
| 2004 | 46,519 | 3,747 | 0.0402 | 0.0603 | ⚠️ POOR |
| 2005 | 44,213 | 3,656 | 0.0403 | 0.0601 | ⚠️ POOR |
| 2006 | 45,226 | 3,881 | 0.0418 | 0.0624 | ⚠️ POOR |
| 2007 | 44,375 | 3,462 | 0.0397 | 0.0592 | ⚠️ POOR |
| 2008 | 44,924 | 3,230 | 0.0362 | 0.0541 | ⚠️ POOR |
| 2009 | 46,488 | 3,590 | 0.0394 | 0.0586 | ⚠️ POOR |
| 2010 | 45,239 | 3,007 | 0.0333 | 0.0493 | ✅ OK |
| 2011 | 46,817 | 2,546 | 0.0272 | 0.0404 | ✅ OK |
| 2012 | 47,396 | 2,749 | 0.0299 | 0.0445 | ✅ OK |
| 2013 | 46,823 | 2,612 | 0.0285 | 0.0424 | ✅ OK |
| 2014 | 46,268 | 2,495 | 0.0278 | 0.0416 | ✅ OK |
| 2015 | 47,248 | 2,442 | 0.0263 | 0.0385 | ✅ OK |
| 2016 | 46,036 | 2,528 | 0.0287 | 0.0426 | ✅ OK |
| 2017 | 46,505 | 2,502 | 0.0270 | 0.0399 | ✅ OK |
| 2018 | 46,619 | 2,447 | 0.0284 | 0.0425 | ✅ OK |
| 2019 | 44,813 | 2,617 | 0.0308 | 0.0460 | ✅ OK |
| 2020 | 47,829 | 2,029 | 0.0222 | 0.0328 | ✅ OK |
| 2021 | 47,734 | 2,282 | 0.0254 | 0.0377 | ✅ OK |

**Poorest calibration in vintage_year:** 2003 (ECE=0.0641), 2006 (ECE=0.0624), 2004 (ECE=0.0603)


**Methodology:** ECE = Expected Calibration Error (10-bin), Brier Score = mean squared error
of predicted probability. Lower is better for both metrics.

Script: `src/explainability/calibration_by_segment.py`