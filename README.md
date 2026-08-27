# Loan Performance Intelligence Engine

> **Intain Campus FinTech Challenge 2026 — AI Track**  
> An ML-first, production-grade platform for loan-level data profiling, multi-outcome performance prediction, survival modeling, anomaly detection, macro stress simulations, explainability, and grounded LLM reviewer assistance.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Loan Performance Intelligence Engine                            │
├───────────────────┬─────────────────────┬────────────────────────┬──────────────────────────────┤
│  Data Generation  │  Data Intelligence  │  Predictive Modeling   │  Survival & Transitions      │
│  (Phase 1)        │  (Task 1)           │  (Task 2)              │  (Task 3)                    │
│                   │                     │                        │                              │
│ • 50k loans panel │ • Column Profiling  │ • LightGBM + LogReg    │ • Kaplan-Meier curves        │
│ • Injected MNAR   │ • MCAR/MNAR Tests   │ • 5 Prediction Targets │ • Cox PH Regression (C=0.71) │
│ • Cross breaks    │ • PSI / KS Drift    │ • Time-Aware Split     │ • 7-State Markov Matrix      │
│ • Servicer feeds  │ • Composite DQ (97) │ • Platt Calibration    │ • Baseline Lift (+0.21)      │
├───────────────────┴─────────────────────┴────────────────────────┴──────────────────────────────┤
│  Anomaly & Exceptions (Task 4) │ Macro Scenarios (Task 5)     │ Explainability (Task 6)        │
│ • Isolation Forest [0, 1]      │ • Base / Adverse / High Prep │ • TreeSHAP Global Rankings     │
│ • Hybrid Rule Classifier       │ • Segment Vulnerability      │ • Local Waterfall per Loan     │
│ • 25 Reviewer Case Studies     │ • 2.4x Subprime Stress Rate  │ • Confidence & Error Audits    │
├────────────────────────────────┴──────────────────────────────┴────────────────────────────────┤
│  Grounded LLM Copilot (Task 7)                                │ Verification & Governance (Task 8)            │
│ • Context Retrieval over Schema, Rules & ML Probabilities      │ • Rigorous Test Suite (46 pytest tests)       │
│ • Verbatim JSONL Audit Logging (logs/llm_prompt_log.jsonl)     │ • Disqualification Self-Audit Compliance      │
│ • Governance: "Recommendation — not a decision."               │ • End-to-End Reproducibility (Makefile)       │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Scale & Time-Aware Partitioning

> **Data Storage & Generation Note**: Raw data CSVs are generated locally and deterministically with fixed seed (`seed=42`) via `make data` (or `python src/data_generation/generate.py`) and are gitignored to keep repository clone size minimal (<3 MB). Running `make data` reproduces the exact 50,000-loan / 944,306-row panel dataset described below in ~20 seconds.

- **Total Population**: **50,000 unique loans** across 20 US states (`data/raw/loan_static_attributes.csv`, 5.08 MB).
- **Historical Training Panel**: **874,435 monthly performance records** across **46,413 unique loans** (originated $\le$ 2021-12 in `data/raw/loan_monthly_performance_train.csv`, 180.05 MB). Split strictly by vintage cohort into:
  - **Train Partition**: **778,872 rows** (41,477 unique loans, originated $\le$ 2019-12)
  - **Out-of-Time Validation Partition**: **95,563 rows** (4,936 unique loans, originated 2020-01 to 2021-12)
- **Held-Out Test Panel**: **69,871 monthly performance records** across **3,587 unique loans** (originated $\ge$ 2022-01 in `data/raw/loan_monthly_performance_test.csv`, 14.40 MB). Zero `loan_id` overlap with training panel (formally asserted by automated tests in `tests/test_schema_validation.py` and `tests/test_splitter.py`).
- **Final Scored Submission**: Exactly **3,587 rows** (`submission/submission.csv`), representing 100% of the unique held-out test loan cohort evaluated at their latest monthly surveillance snapshot.

---

## 3. Quickstart & Reproducibility

### Setup Virtual Environment
```bash
git clone https://github.com/ac265640/LoanScope.git
cd LoanScope
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY or OPENAI_API_KEY (optional)
```

### Run Full End-to-End Pipeline
```bash
make run-all
```
*(Approximate runtime: ~2.5 minutes end-to-end on a standard developer laptop)*

Or execute task by task:
```bash
make data            # Phase 1: Generate synthetic 50,000 loans panel dataset
make test            # Run 46 automated pytest tests (leakage, schema, rolling features, parity)
make profile         # Task 1: Generate Data Intelligence report
make train           # Task 2: Train baseline & LightGBM prediction models
make thresholds      # Task 2b: Optimize F1 classification thresholds
make survival        # Task 3: Fit Kaplan-Meier, Cox PH & Markov transition matrix
make competing-risks # Task 3b: Fit Cause-Specific Cumulative Incidence Functions (CIF)
make anomaly         # Task 4: Run Isolation Forest & generate 25 reviewer cases
make scenarios       # Task 5: Run Base, Adverse Credit & High Prepayment stress
make monte-carlo     # Task 5b: Run 1,000-path Monte Carlo portfolio simulations
make explain         # Task 6: Compute SHAP values, error audit, and Model Card
make fairness        # Task 6b: Conduct Algorithmic Fairness & Disparate Impact audit
make copilot         # Task 7: Execute grounded batch LLM reviewer copilot demo
make drift           # Task 7b: Run Feature Drift Monitoring dashboard (PSI + KS)
make submission      # Generate final submission/submission.csv
```

### Unified CLI Runner
```bash
python src/pipeline/cli.py all           # Run complete pipeline
python src/pipeline/cli.py drift         # Run drift monitor
python src/pipeline/cli.py fairness      # Run fairness audit
python src/pipeline/cli.py monte-carlo   # Run Monte Carlo stress
```

---

## 4. Quantitative Model Results

### Task 2: Predictive Modeling Performance (Out-of-Time Validation Cohort)

| Target Variable | Baseline (LogReg) AUC | Improved (LightGBM) AUC | Naive Prev. PR-AUC | Baseline PR-AUC | Improved LightGBM PR-AUC | Brier Score | Precision @ Top 5% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `next_3m_delinquency_flag` | 0.7780 | **0.7977** (+0.0197) | 0.0404 | 0.2683 | **0.4090** (+0.1408, 10.1x prev) | **0.0291** | **33.82%** (8.4x lift) |
| `next_6m_delinquency_flag` | 0.7464 | **0.7656** (+0.0192) | 0.0627 | 0.2607 | **0.3599** (+0.0992, 5.7x prev) | **0.0486** | **39.66%** (6.3x lift) |
| `next_12m_default_flag` | 0.7008 | **0.7179** (+0.0171) | 0.0451 | 0.1103 | **0.1401** (+0.0298, 3.1x prev) | **0.0415** | **18.52%** (4.1x lift) |
| `next_12m_prepayment_flag` | 0.6773 | **0.6738** (-0.0035) | 0.0470 | 0.0828 | **0.0816** (-0.0011, 1.7x prev) | **0.0442** | **10.69%** (2.3x lift) |
| `next_state` (6-class) | N/A | **N/A** | N/A | Macro-F1: 0.5111 | **Macro-F1: 0.5432** (+0.0321) | N/A | Top-1 Acc: **84.2%** |

### Task 4: Anomaly & Exception Classification (Separated by Component)
- **Component A (Deterministic Rule Engine)**: Evaluates hard constraints (VR001–VR005) with **100.00% Rule Match Rate** (by construction).
- **Component B (Learned ML Model)**: Non-circular LightGBM on 32 behavioral features achieves **ROC-AUC: 0.8310**, **F1 @ 0.50: 0.7361**, and **Macro-F1: 0.5914**.

*Note: All models use time-aware cohort splits by `origination_month` with formally asserted zero `loan_id` data leakage.*

---

## 5. Key Deliverables Directory

| Deliverable | File Location | Description |
| :--- | :--- | :--- |
| **Final Submission File** | [`submission/submission.csv`](submission/submission.csv) | Final scored test dataset with probabilities, anomaly scores, top drivers, and actions. |
| **Data Intelligence Report** | [`reports/data_intelligence_report.md`](reports/data_intelligence_report.md) | Full distributions, missingness (MNAR/MCAR), outliers, drift, and DQ scores. |
| **Model Card** | [`reports/model_card.md`](reports/model_card.md) | Official model specifications, data lineage, validation method, and limitations. |
| **Explainability Report** | [`reports/explainability_report.md`](reports/explainability_report.md) | Global SHAP importance, local waterfalls, uncertainty, and FP/FN audits. |
| **Scenario Stress Report** | [`reports/scenario_report.md`](reports/scenario_report.md) | Macro simulations (Base, Adverse Credit, High Prepayment) and segment curves. |
| **Anomaly Reviewer Cases** | [`reports/anomaly_reviewer_cases.md`](reports/anomaly_reviewer_cases.md) | 25 detailed reviewer cases with plain-English diagnostic explanations. |
| **LLM Prompt Log** | [`logs/llm_prompt_log.jsonl`](logs/llm_prompt_log.jsonl) | Verbatim audit log of all grounded copilot prompt payloads and responses. |
| **LLM Hallucination Audit** | [`reports/llm_hallucination_audit.md`](reports/llm_hallucination_audit.md) | Grounded copilot deterministic guardrails and hallucination interception audit. |
| **Walkthrough Notebook** | [`notebooks/end_to_end_walkthrough.ipynb`](notebooks/end_to_end_walkthrough.ipynb) | End-to-end reproducible walkthrough notebook across all 8 tasks. |
| **Fairness & Governance Report** | [`reports/fairness_report.md`](reports/fairness_report.md) | Subgroup disparate impact ratios, four-fifths rule compliance, and error parity audits. |

---

## 6. Disqualification Self-Audit Status

- [x] **No LLM-only prediction** (All predictions generated by non-LLM models: LightGBM, LogisticRegression, Cox PH, Isolation Forest).
- [x] **Trained non-LLM models saved** (`.joblib` binaries in `src/models/saved_models/`).
- [x] **Time-aware split with zero `loan_id` leakage** (Asserted via `pytest tests/test_splitter.py`).
- [x] **Zero target leakage into features** (32 features calculated strictly from backward-looking historical observations).
- [x] **Reproducible code and evaluation metrics** (Full `Makefile` with fixed seed `42`).
- [x] **Grounded LLM reviewer summaries** (Grounded retrieval, verbatim prompt logs, labeled as advisory recommendation).

---

## 7. Advanced Features Suite (All 15 Implemented & Audited)

Every advanced feature listed in Section 10 of the problem statement is fully functional, producing real quantitative outputs from actual data and models:

| # | Advanced Feature | Description | Executable Module | Output Artifact / Report |
|---|------------------|-------------|-------------------|--------------------------|
| **1** | **Competing-Risk Survival Model** | Cause-specific Aalen-Johansen CIF curves for default vs prepayment with credit band cuts and single-risk bias quantification | `src/models/survival/competing_risk.py` | [`reports/survival_report.md`](reports/survival_report.md) |
| **2** | **Monte Carlo Portfolio Simulation** | 1,000-path stochastic portfolio stress simulation drawing from Beta uncertainty distributions (P1–P99 fan intervals) | `src/scenarios/monte_carlo.py` | [`reports/scenario_report.md`](reports/scenario_report.md) |
| **3** | **Drift Monitoring Dashboard** | Interactive Streamlit & Plotly app computing PSI and KS statistics between train and test splits with pass/warn/fail thresholds | `src/monitoring/drift_dashboard.py` | [`reports/drift_monitoring_report.md`](reports/drift_monitoring_report.md) |
| **4** | **Segment-Level Scenario Curves** | Time-series stress projection curves across 5,400 segment-month combinations sliced by vintage, credit band, state, and servicer | `src/scenarios/segment_curves.py` | [`reports/scenario_report.md`](reports/scenario_report.md) |
| **5** | **Model Calibration by Segment** | Separate 10-bin reliability diagrams, Expected Calibration Error (ECE), and Brier scores across credit bands and vintage eras | `src/explainability/calibration_by_segment.py` | [`reports/calibration_by_segment_report.md`](reports/calibration_by_segment_report.md) |
| **6** | **MLflow Experiment Tracking** | Multi-task experiment tracking logging parameters, metrics, and models across predictive, survival, and anomaly model families | `src/pipeline/mlflow_tracking.py` | `mlruns/` & `logs/mlruns/` |
| **7** | **RAG over Data Dictionary & Rules** | BM25/TF-IDF chunked retrieval grounding LLM prompt queries with verbatim chunk logging to prompt log | `src/llm_copilot/rag.py` | [`logs/llm_prompt_log.jsonl`](logs/llm_prompt_log.jsonl) |
| **8** | **Agentic Experiment Runner** | Autonomous orchestrator sweeping model architectures and feature subsets with best-first heuristic next-configuration proposals | `src/pipeline/experiment_runner.py` | [`configs/sweep.yaml`](configs/sweep.yaml) & [`logs/sweep_results.json`](logs/sweep_results.json) |
| **9** | **Automated Feature-Store Pipeline** | Versioned feature-store computing registered features with schema manifests, registry definitions, and parquet caching | `src/features/feature_store.py` | [`data/processed/feature_store/registry.json`](data/processed/feature_store/registry.json) |
| **10** | **Bias / Fairness Analysis** | Disparate impact ratio evaluation, four-fifths rule compliance, and error rate analysis across lending proxy segments | `src/explainability/fairness_analysis.py` | [`reports/fairness_report.md`](reports/fairness_report.md) |
| **11** | **Counterfactual Explanations** | Loan-level perturbation-and-rescore generating actionable feature levers ("if credit band rose to 700, default prob drops by 6%") | `src/explainability/counterfactuals.py` | [`reports/counterfactuals.json`](reports/counterfactuals.json) & [`reports/explainability_report.md`](reports/explainability_report.md) |
| **12** | **Stress Sensitivity by Feature Cluster** | Attribution decomposition identifying which feature cluster (credit quality, rates, loan size, geography) drives scenario default rates | `src/scenarios/stress_sensitivity.py` | [`reports/scenario_report.md`](reports/scenario_report.md) |
| **13** | **Model Confidence Intervals** | Split conformal prediction intervals providing finite-sample guaranteed marginal coverage (90.3% empirical coverage @ 90% target) | `src/explainability/confidence_intervals.py` | [`reports/confidence_intervals_report.md`](reports/confidence_intervals_report.md) |
| **14** | **Human-in-the-Loop Active Learning** | Reviewer accept/reject/correct feedback loop recalibrating anomaly detection thresholds with +8.2pp precision improvement | `src/models/anomaly/active_learning.py` | [`reports/active_learning_report.md`](reports/active_learning_report.md) |
| **15** | **Synthetic-Data Stress Testing** | Edge-case validation testing graceful pipeline degradation against simulated 2008 recession cohorts and severe DQ corruption batches | `src/data_generation/stress_test_data.py` | [`reports/stress_test_report.md`](reports/stress_test_report.md) |

---

## 8. License

This project is licensed under the MIT License — see the [`LICENSE`](LICENSE) file for details.

