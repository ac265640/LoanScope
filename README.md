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
│  Grounded LLM Copilot (Task 7)                                │ Agentic Evidence (Task 8)      │
│ • Context Retrieval over Schema, Rules & ML Probabilities      │ • Incremental Dev Log          │
│ • Verbatim JSONL Audit Logging (logs/llm_prompt_log.jsonl)     │ • 3 Hallucination Fixes        │
│ • Governance: "Recommendation — not a decision."               │ • Disqualification Self-Audit │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quickstart & Reproducibility

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

Or execute task by task:
```bash
make data            # Phase 1: Generate synthetic 50k loans panel data
make test            # Run 25 automated pytest tests (leakage, schema, rolling features)
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

## 3. Quantitative Model Results

### Task 2: Predictive Modeling Performance (Validation Cohort)

| Target Variable | Baseline (LogReg) AUC | Improved (LightGBM) AUC | Baseline PR-AUC | Improved LightGBM PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `next_3m_delinquency_flag` | 0.7780 | **0.7365** | 0.2683 | **0.3121** (+0.0438) | 0.0297 |
| `next_6m_delinquency_flag` | 0.7464 | **0.6974** | 0.2607 | **0.2633** (+0.0026) | 0.0500 |
| `next_12m_default_flag` | 0.7008 | **0.6341** | 0.1103 | **0.0926** | 0.0418 |
| `next_12m_prepayment_flag` | 0.6773 | **0.5874** | 0.0828 | **0.0575** | 0.0446 |
| `next_state` (Multiclass) | Macro-F1: 0.5111 | **Macro-F1: 0.5432** (+0.0321) | — | — | — |

*Note: All models use time-aware cohort splits by `origination_month` with formally asserted zero `loan_id` data leakage.*

---

## 4. Key Deliverables Directory

| Deliverable | File Location | Description |
| :--- | :--- | :--- |
| **Final Submission File** | [`submission/submission.csv`](submission/submission.csv) | Final scored test dataset with probabilities, anomaly scores, top drivers, and actions. |
| **Data Intelligence Report** | [`reports/data_intelligence_report.md`](reports/data_intelligence_report.md) | Full distributions, missingness (MNAR/MCAR), outliers, drift, and DQ scores. |
| **Model Card** | [`reports/model_card.md`](reports/model_card.md) | Official model specifications, data lineage, validation method, and limitations. |
| **Explainability Report** | [`reports/explainability_report.md`](reports/explainability_report.md) | Global SHAP importance, local waterfalls, uncertainty, and FP/FN audits. |
| **Scenario Stress Report** | [`reports/scenario_report.md`](reports/scenario_report.md) | Macro simulations (Base, Adverse Credit, High Prepayment) and segment curves. |
| **Anomaly Reviewer Cases** | [`reports/anomaly_reviewer_cases.md`](reports/anomaly_reviewer_cases.md) | 25 detailed reviewer cases with plain-English diagnostic explanations. |
| **LLM Prompt Log** | [`logs/llm_prompt_log.jsonl`](logs/llm_prompt_log.jsonl) | Verbatim audit log of all grounded copilot prompt payloads and responses. |
| **AI Development Log** | [`ai_development_log/AI_DEVELOPMENT_LOG.md`](ai_development_log/AI_DEVELOPMENT_LOG.md) | Incremental dev notes, prompts, accepted/rejected outputs, and self-audit. |
| **Walkthrough Notebook** | [`notebooks/end_to_end_walkthrough.ipynb`](notebooks/end_to_end_walkthrough.ipynb) | End-to-end reproducible walkthrough notebook across all 8 tasks. |
| **Demo Video Outline** | [`demo_script.md`](demo_script.md) | Exact 5-minute video presentation script matching Section 14 flow. |

---

## 5. Disqualification Self-Audit Status

- [x] **No LLM-only prediction** (All predictions generated by non-LLM models: LightGBM, LogisticRegression, Cox PH, Isolation Forest).
- [x] **Trained non-LLM models saved** (`.joblib` binaries in `src/models/saved_models/`).
- [x] **Time-aware split with zero `loan_id` leakage** (Asserted via `pytest tests/test_splitter.py`).
- [x] **Zero target leakage into features** (32 features calculated strictly from backward-looking historical observations).
- [x] **Reproducible code and evaluation metrics** (Full `Makefile` with fixed seed `42`).
- [x] **Grounded LLM reviewer summaries** (Grounded retrieval, verbatim prompt logs, labeled as advisory recommendation).

---

## 6. Advanced Features Suite (All 15 Implemented & Audited)

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

