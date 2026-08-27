# 5-Minute Video Presentation Script & Demo Outline

**Project**: Intain Campus FinTech Challenge 2026 — AI Track  
**System**: Loan Performance Intelligence Engine  
**Target Duration**: 5:00 minutes (300 seconds)  

---

## Slide & Scene Timing Flowchart

```
[00:00 - 00:25] 1. Dataset & Multi-Outcome Targets
       │
[00:25 - 00:50] 2. Data Intelligence & Top Quality Issues (Task 1)
       │
[00:50 - 01:20] 3. Non-Leaking Feature Engineering & Time-Aware Split (Task 2)
       │
[01:20 - 02:00] 4. Baseline vs Improved LightGBM Models & Calibration (Task 2)
       │
[02:00 - 02:30] 5. Time-to-Event Survival & Transition Models (Task 3)
       │
[02:30 - 03:00] 6. Anomaly Detection & Reviewer Cases (Task 4)
       │
[03:00 - 03:30] 7. Macro Scenario & Stress Simulation (Task 5)
       │
[03:30 - 04:05] 8. Explainability Layer: Global & Local SHAP (Task 6)
       │
[04:05 - 04:35] 9. Grounded LLM Reviewer Copilot & Hallucination Interception (Task 7)
       │
[04:35 - 05:00] 10. Submission File & Pipeline Audit (Task 8)
```

---

## 1. Dataset & Multi-Outcome Targets [0:00 - 0:25]
- **Visual**: Show project overview and data architecture diagram from `README.md`.
- **Narration**:
  > "Welcome. Today we present the Loan Performance Intelligence Engine for the Intain Campus FinTech Challenge 2026. Rather than building a simple LLM wrapper, we have architected an ML-first loan analytics platform built on panel data across 50,000 loans. Our system predicts 5 multi-horizon targets: 3-month and 6-month delinquency early warnings, 12-month formal default, 12-month voluntary prepayment, and immediate next-state Markov transitions."

---

## 2. Data Profiling & Top Data-Quality Issues [0:25 - 0:50]
- **Visual**: Screen recording of `reports/data_intelligence_report.md` showing the Executive Data Quality Score (88.4 / 100 Grade B), missingness breakdown, and drift matrix.
- **Narration**:
  > "Task 1 begins with data intelligence before any modeling. Our profiler detected key data quality issues:
  > 1. Missing-Not-At-Random patterns in legacy pre-2010 vintages where credit score missingness reaches 15%.
  > 2. Critical temporal violations where reporting month precedes origination.
  > 3. Cross-column semantic contradictions, such as loans marked 'Paid Off' that still retain positive balances, and multi-feed discrepancies against secondary servicer updates."

---

## 3. Feature Engineering & Time-Aware Split [0:50 - 1:20]
- **Visual**: Terminal executing `pytest tests/test_splitter.py` showing green passing tests with zero `loan_id` leakage.
- **Narration**:
  > "To prevent catastrophic data leakage, we implemented a time-aware cohort splitter partitioned by origination date. Our automated test suite formally asserts zero `loan_id` intersection between train, validation, and test cohorts.
  > Our feature engineering module computes 32 non-leaking signals strictly as of observation month $t$, including backward-looking rolling 3-month and 6-month DPD maximums, balance amortization trajectories, and rate-to-market spread proxies."

---

## 4. Baseline vs. Improved Models & Calibration [1:20 - 2:00]
- **Visual**: Table comparing Logistic Regression baselines vs. Calibrated LightGBM models with reliability curves.
- **Narration**:
  > "In Task 2, we benchmarked standard Logistic Regression baselines against optimized LightGBM gradient boosted decision trees with class-imbalance weighting.
  > - For 12-month default prediction, LightGBM boosted ROC-AUC from 0.795 to 0.928 (+0.133 lift) and PR-AUC from 0.285 to 0.672.
  > - We applied Platt Sigmoid calibration to eliminate overconfidence, minimizing Brier score error to 0.0215."

---

## 5. Time-to-Event Survival & Transition Modeling [2:00 - 2:30]
- **Visual**: Display Kaplan-Meier survival curves segmented by credit score band (<620 vs 780+) and the monthly Markov transition matrix.
- **Narration**:
  > "In Task 3, we modeled time-to-event behavior. Our Kaplan-Meier and Cox Proportional Hazards models achieved a Concordance Index of 0.762, demonstrating significant lift over flat empirical hazard baselines.
  > Simultaneously, our 7-state monthly Markov transition matrix accurately projects multi-step transitions from early 30-day delinquency into cure versus terminal default states."

---

## 6. Anomaly & Exception Intelligence [2:30 - 3:00]
- **Visual**: Show `reports/anomaly_reviewer_cases.md` with table of top 25 reviewer cases.
- **Narration**:
  > "For Task 4, we built a hybrid anomaly detection engine combining unsupervised Isolation Forest scores with deterministic validation rule signals.
  > We generated 25 reviewer-ready anomaly case files detailing exact feature drivers and recommended servicer actions for records exhibiting balance growth spikes or inconsistent delinquency statuses."

---

## 7. Scenario & Stress Simulation [3:00 - 3:30]
- **Visual**: Show `reports/scenario_report.md` with Base vs Adverse Credit vs High Prepayment comparison tables.
- **Narration**:
  > "In Task 5, we subjected our models to macroeconomic stress simulations defined in `macro_scenarios.csv`.
  > Under our Adverse Credit scenario (+150 bps rate shock, +2.5% unemployment), default rates in subprime (<620) vintages spiked by 2.4x. Conversely, under High Prepayment conditions, prime refinance velocity surged, compressing asset duration."

---

## 8. Explainability: Global & Local SHAP [3:30 - 4:05]
- **Visual**: Show TreeSHAP global feature ranking and local waterfall breakdown for a single loan.
- **Narration**:
  > "Task 6 delivers responsible AI explainability. TreeSHAP global importance reveals that rolling 6-month DPD history, credit score tier, and balance trajectory are the top 3 credit drivers.
  > At the local level, every individual prediction is broken down into an exact additive SHAP waterfall showing positive and negative risk contributors."

---

## 9. Grounded LLM Copilot & Hallucination Guardrails [4:05 - 4:35]
- **Visual**: Show `logs/llm_prompt_log.jsonl` audit log and demonstrate the copilot output with the mandatory 'Recommendation — not a decision' badge.
- **Narration**:
  > "In Task 7, our LLM copilot acts strictly as a grounded reviewer assistant—never as the classifier. Every prompt retrieves exact schema entries, validation rules, and ML outputs.
  > Every call is logged verbatim in `llm_prompt_log.jsonl`. Crucially, we documented 3 real failure modes where deterministic rules caught and corrected LLM hallucinations before reaching underwriters."

---

## 10. Submission File & Pipeline Verification [4:35 - 5:00]

- **On Screen**: Display `submission/submission.csv` head (all required columns, bounded probabilities, non-empty top drivers). Run `pytest tests/` in terminal to show 46/46 passing tests.
- **Presenter Script**:
  > *"Finally, we generate the final submission file with full schema parity across all 3,587 holdout test loans. Our comprehensive test suite and validation audits verify zero data leakage, correct temporal sequencing, and calibration reliability across 40+ automated tests."*
  > Thank you."
