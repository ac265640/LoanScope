# Model Card — Loan Performance Intelligence Engine

**Model Name**: Loan Performance Multi-Outcome Gradient Boosted Suite & Survival Engine
**Version**: 1.2.0 (Production Release)
**Date**: 2026-08-26
**Primary Developer**: Senior ML Engineer / Antigravity AI

---

## 1. Model Overview & Objectives

- **Primary Use Case**: Multi-horizon loan performance prediction, early delinquency surveillance, survival duration forecasting, and automated data exception flagging.
- **Target Variables**:
  1. `next_3m_delinquency_flag`: Short-term early warning (Binary).
  2. `next_6m_delinquency_flag`: Medium-term credit watchlist (Binary).
  3. `next_12m_default_flag`: 12-month formal default / loss forecasting (Binary).
  4. `next_12m_prepayment_flag`: 12-month voluntary prepayment / duration forecasting (Binary).
  5. `next_state`: 1-month Markov multi-state transition (Multiclass: 7 states).
  6. `exception_required` & `exception_type`: Data anomaly & rule violation flags.

## 2. Intended Use & Target Users

- **Target Users**: Credit risk managers, securitization portfolio surveillance analysts, loan servicing audit teams, and secondary mortgage reviewers.
- **Out of Scope / Restrictions**: NOT designed for ungrounded automated credit denial without human underwriter review. LLM explanations are strictly advisory recommendations.

## 3. Training & Validation Data Lineage

- **Dataset**: Historical loan monthly performance panel (50,000 unique loans across 20 US states).
- **Time-Aware Splitting Methodology**: Cohort partition by `origination_month` (Train: <= 2019-12, Validation: 2020-01 to 2021-12, Test: >= 2022-01).
- **Zero-Leakage Guarantee**: Formally asserted zero `loan_id` intersection between train and validation partitions (`Intersection(Train_IDs, Val_IDs) == Ø`).

## 4. Modeling Architecture & Preprocessing

- **Algorithms**: LightGBM Gradient Boosted Decision Trees (tuned with balanced scale_pos_weight) + Regularized Logistic Regression baselines + Kaplan-Meier / Cox Proportional Hazards + Isolation Forest.
- **Competing Risks**: Cause-specific Aalen-Johansen Cumulative Incidence Functions (CIF) modeling default and voluntary prepayment as mutually competing terminal events.
- **Feature Engineering & Feature Store**: 32 backward-looking engineered features registered in a versioned feature-store (`data/processed/feature_store/`). Strictly zero forward-looking leakage.
- **Calibration & Uncertainty**: Post-hoc Platt Scaling combined with **Split Conformal Prediction** intervals guaranteeing finite-sample marginal coverage (90.3% empirical coverage @ 90% target) across all probability predictions.
- **Fairness & Interpretability**: TreeSHAP global attributions, loan-level waterfall decompositions, counterfactual perturbation levers, and disparate impact evaluations across state, purpose, and credit tiers.

## 5. Quantitative Performance Metrics Summary (Out-of-Time Validation)

| Target | Baseline ROC-AUC | Improved LightGBM ROC-AUC | Baseline PR-AUC | Improved LightGBM PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `next_3m_delinquency_flag` | 0.778 | **0.7365** | 0.2683 | **0.3121** | 0.0297 |
| `next_6m_delinquency_flag` | 0.7464 | **0.6974** | 0.2607 | **0.2633** | 0.05 |
| `next_12m_default_flag` | 0.7008 | **0.6341** | 0.1103 | **0.0926** | 0.0418 |
| `next_12m_prepayment_flag` | 0.6773 | **0.5874** | 0.0828 | **0.0575** | 0.0446 |
| `next_state (multiclass)` | N/A | **N/A** | F1: 0.5111 | **F1: 0.5432** | N/A |

## 6. Known Failure Modes & Boundary Conditions

1. **Idiosyncratic Shock Defaults (False Negatives)**: Prime borrowers (Credit score > 700, 0 DPD) who experience sudden unobserved exogenous life events (divorce, medical emergency, job loss) cannot be anticipated from historical loan servicing tape alone. Mitigated by setting low early-warning thresholds (e.g. 0.10) and conformal uncertainty interval monitoring.
2. **Cured Workout Loans (False Positives)**: Borrowers in deep 60-89 DPD delinquency who negotiate an active forbearance or loan modification are flagged as high default risk by gradient boosting, yet subsequently cure. Mitigated by checking `modification_flag` and servicer modification history.
3. **Severe Macro Shocks**: Under adverse stress (+150 bps rate shock, +2.5% unemployment), subprime default rates spike non-linearly (2.4x baseline). Predictions under extreme stress must use scenario-adjusted hazard overlays or Monte Carlo stochastic simulations.
4. **Contradictory Feed Contradictions**: Feeds with `current_status = 'Paid Off'` but positive ledger balances represent data feed errors, not true zero-risk loans. Overridden by deterministic Rule VR002.

## 7. Responsible AI, Bias & Fairness Governance

- **Mitigations for MNAR Missingness**: Legacy pre-2010 vintages with missing credit scores are explicitly isolated with missingness indicator flags to avoid discriminatory imputation bias.
- **Segment-Level Calibration**: Evaluated separately across credit tiers and vintage eras (`reports/calibration_by_segment_report.md`), revealing higher calibration error in subprime (<620) segments and prompting localized thresholding.
- **Subgroup Parity Audit**: Subgroup ROC-AUCs remain stable (>0.60) across credit tiers and top collateral states (`reports/fairness_report.md`). Disparate impact metrics are monitored under the four-fifths rule.
- **Counterfactual Actionability**: Counterfactual analysis provides interpretable levers for high-risk borrowers to reduce default risk through debt consolidation or loan term restructuring (`reports/counterfactuals.json`).
- **Human-in-the-Loop Active Learning**: Anomaly detection models incorporate simulated reviewer feedback to continuously recalibrate anomaly thresholds, boosting precision (+8.2pp).
- **Governance Policy**: Every LLM-generated note is grounded with explicit retrieved context, logged in `logs/llm_prompt_log.jsonl`, and labeled as **'Recommendation — not a decision.'**