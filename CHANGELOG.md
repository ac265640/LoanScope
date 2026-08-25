# Changelog

All notable changes to the **Loan Performance Intelligence Engine** will be documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-08-26
### Added
- **Algorithmic Fairness Audit**: Subgroup ROC-AUC, Demographic Parity Ratio (DPR), and False Positive Rate (FPR) parity evaluation across credit bands and collateral states (`src/explainability/fairness_audit.py`).
- **Competing-Risk Survival Modeling**: Cause-specific Nelson-Aalen cumulative incidence function (CIF) estimates for default vs. prepayment (`src/models/survival/competing_risks.py`).
- **Monte Carlo Stress Engine**: 1,000-path stochastic portfolio simulations generating P5/P50/P95 confidence intervals under Base, Adverse Credit, and High Prepayment shocks (`src/scenarios/monte_carlo.py`).
- **Classification Threshold Optimizer**: Precision-Recall curve sweep maximizing F1-score across all imbalanced binary targets (`src/models/prediction/threshold_optimizer.py`).
- **RAG Knowledge Retriever**: BM25 token-overlap retrieval over `data_dictionary.md` and `validation_rules.json` (`src/llm_copilot/rag_retriever.py`).
- **Segment-Level Calibration**: Expected Calibration Error (ECE) and Brier scores disaggregated by credit tier and vintage era (`src/models/prediction/segment_calibration.py`).
- **Unified CLI Entrypoint**: CLI runner orchestrating individual pipeline stages (`src/pipeline/cli.py`).
- **Comprehensive Pytest Suite**: 25 unit tests covering time-aware splitting, schema validation, data invariants, and rolling feature calculations (`tests/`).

---

## [1.1.0] - 2026-08-25
### Added
- **Feature Drift Monitoring**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) distributional drift dashboard across all 32 engineered features (`src/pipeline/drift_monitor.py`).
- **MLflow Experiment Tracking**: Versioned tracking of hyperparameters, classification metrics, feature importances, and model binaries (`src/pipeline/mlflow_tracking.py`).
- **Batch Grounded Reviewer Copilot**: 10-profile diverse reviewer generator with verbatim logging to `logs/llm_prompt_log.jsonl` (`src/llm_copilot/batch_copilot.py`).
- **Hallucination Interception Suite**: Documented 3 failure cases and automated deterministic rule overrides (`src/llm_copilot/hallucination_cases.py`).

---

## [1.0.0] - 2026-08-25
### Added
- **Phase 1 Synthetic Generator**: 50,000 loans × up to 36 months panel dataset (~944k rows) with injected MNAR/MCAR missingness, date anomalies, and multi-feed discrepancies.
- **Task 1 Data Intelligence Suite**: Comprehensive profilers for distributions, missingness, Tukey/Isolation Forest outliers, correlation, relationship breaks, and composite Data Quality scoring (`src/profiling/`).
- **Task 2 Predictive Modeling**: Non-leaking backward-looking features, time-aware cohort splitter with asserted zero `loan_id` leakage, LightGBM models, and Platt probability calibration (`src/models/prediction/`).
- **Task 3 Survival Modeling**: Segmented Kaplan-Meier curves, Cox Proportional Hazards regression (C-index = 0.7059), and 7-state Markov transition matrix (`src/models/survival/`).
- **Task 4 Anomaly & Exception Intelligence**: Isolation Forest scoring, hybrid deterministic rule classifier, and 25 detailed reviewer cases (`src/models/anomaly/`).
- **Task 5 Scenario Simulation**: Base, Adverse Credit (+150 bps rate shock, +2.5% unemployment), and High Prepayment (-75 bps) portfolio and segment projections (`src/scenarios/scenario_runner.py`).
- **Task 6 Explainability**: TreeSHAP global feature rankings, local waterfall attributions, confidence scores, FP/FN error analysis, and Model Card (`src/explainability/`).
- **Task 7 Grounded LLM Copilot**: Context retriever, reviewer notes with mandatory "Recommendation — not a decision." label (`src/llm_copilot/`).
- **Task 8 Agentic Evidence & Submission**: Complete `AI_DEVELOPMENT_LOG.md`, reproducible walkthrough notebook, 5-minute video demo script, and final `submission/submission.csv`.
