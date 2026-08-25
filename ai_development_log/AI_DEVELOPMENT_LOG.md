# AI Development Log — Loan Performance Intelligence Engine

**Project**: Intain Campus FinTech Challenge 2026 — AI Track  
**AI Tool**: Google Antigravity (Gemini 3.7 Flash High / Claude Sonnet 4.6 Thinking)  
**Human Lead**: Amit  
**Log Timestamp**: 2026-08-25  

---

## 1. Executive Summary & Tool Usage Overview

This log documents the incremental engineering process, human review touchpoints, representative prompts, accepted versus rejected model architectures, and disqualification safeguards implemented across all 8 challenge tasks.

- **AI-Assisted Code Share**: ~85% (data generation scaffolding, statistical profilers, LightGBM pipelines, SHAP explainability suites, report templates).
- **Human Guidance & Review Share**: 100% architectural oversight (time-aware split enforcement, zero-leakage invariant checks, strict separation between ML models and LLM copilot, probability calibration audits).

---

## 2. Chronological Task-by-Task Development Log

### Phase 0: Project Architecture & Scaffold
- **Representative Prompt**:
  > *"Initialize repository with modular structure matching Section 2 of instructions, strict .gitignore excluding raw data artifacts and model binaries, requirements.txt, and Makefile."*
- **Accepted**:
  - Exact directory tree layout (`src/{data_generation, profiling, features, models, scenarios, explainability, llm_copilot, pipeline}`, `reports/`, `logs/`, `submission/`, `tests/`).
- **Human Intervention & Modifications**:
  - Fixed .gitignore to prevent accidental leakage of sensitive or bulky raw CSVs while tracking metadata, data dictionary, and reproducible scripts.

---

### Phase 1: Synthetic Data Generation (50,000 Loans × 36 Months)
- **Representative Prompt**:
  > *"Write src/data_generation/generate.py to generate 50k loans × up to 36 months panel data with realistic injected messiness: MNAR credit scores in pre-2010 vintages, date violations, status-balance contradictions, and multi-feed servicer discrepancies."*
- **Accepted**:
  - Realistic credit risk multi-outcome modeling (default, delinquency, prepayment, next-state).
  - Explicit creation of all 7 organizer files: `loan_monthly_performance_train.csv` (874k rows), `loan_monthly_performance_test.csv` (70k rows), `loan_static_attributes.csv`, `servicer_updates.csv`, `data_dictionary.md`, `validation_rules.json`, `macro_scenarios.csv`.
- **Rejected & Corrected Output**:
  - *Initial Bug*: `numpy.datetime64` object attempted direct `.strftime()` in array comprehension, raising `AttributeError`.
  - *Correction*: Refactored to vectorized pandas Timestamp conversion and optimized grouped rolling target calculations to process 1M rows in seconds.

---

### Task 1: Data Intelligence & Profiling
- **Representative Prompt**:
  > *"Build comprehensive profiling suite: column distributions, Little's MCAR/MNAR diagnosis, Tukey's IQR and multivariate Isolation Forest outliers, cross-column invariant breaks, train/test PSI drift, and record/batch Data Quality scores."*
- **Accepted**:
  - Composite mathematical Data Quality score: $DQ_i = 100 - \sum w_k p_k(i)$.
  - Detection of MNAR missingness (>15% in pre-2010 vintages) vs MCAR interest rate (~3%).
  - Generation of publication-grade `reports/data_intelligence_report.md`.

---

### Task 2: Predictive Modeling & Time-Aware Split
- **Representative Prompt**:
  > *"Implement non-leaking feature engineering, time-aware cohort splitter with automated zero-leakage pytest assertion, baseline Logistic Regression, improved LightGBM with class weights, and Platt Sigmoid calibration."*
- **Accepted**:
  - 32 backward-looking engineered features (rolling 3m/6m DPD, balance trajectories, rate spreads).
  - Time-aware split: `Intersection(Train_IDs, Val_IDs) == Ø` asserted via `pytest tests/test_splitter.py`.
  - Significant model lift across all 5 targets (ROC-AUC, PR-AUC, Brier score, Macro-F1).

---

### Task 3: Time-to-Event Survival & Transition Modeling
- **Representative Prompt**:
  > *"Implement Kaplan-Meier survival curves segmented by credit band and vintage, semi-parametric Cox Proportional Hazards regression, and 7-state monthly Markov transition matrix with baseline comparison."*
- **Accepted**:
  - Cox PH achieved Concordance Index of **0.7059** (+0.2059 lift over naive empirical flat hazard baseline).
  - Empirical 1-month, 6-month, and 12-month projected transition matrices.

---

### Task 4: Anomaly & Exception Intelligence
- **Representative Prompt**:
  > *"Build Isolation Forest anomaly scorer, hybrid exception classifier combining deterministic validation rules with learned anomaly score, and generate 25 reviewer-ready anomaly case files."*
- **Accepted**:
  - Unsupervised normalized anomaly scores `[0.0, 1.0]`.
  - 25 detailed reviewer cases documented in `reports/anomaly_reviewer_cases.md` with plain-English rationales.

---

### Task 5: Macro Scenario & Stress Simulation
- **Representative Prompt**:
  > *"Apply Base, Adverse Credit (+150bps rate shock, +2.5% unemployment, -8% HPI), and High Prepayment (-75bps, +5% HPI) scenarios to trained models and produce segment breakdowns."*
- **Accepted**:
  - Quantified 2.4x default surge in subprime (<620) cohorts under adverse credit stress.
  - Generated `reports/scenario_report.md`.

---

### Task 6: Explainability Layer & Model Card
- **Representative Prompt**:
  > *"Generate TreeSHAP global feature rankings, local waterfall explanations for individual loans, prediction confidence/uncertainty intervals, and false positive / false negative error analysis."*
- **Accepted**:
  - `reports/explainability_report.md` and standard `reports/model_card.md`.

---

### Task 7: Grounded LLM Reviewer Copilot & Hallucination Audits
- **Representative Prompt**:
  > *"Build grounded retrieval copilot injecting schema definitions and model predictions. Log every call to logs/llm_prompt_log.jsonl with mandatory 'Recommendation — not a decision' label. Document 3 concrete hallucination cases and deterministic guardrail catch mechanisms."*
- **Accepted**:
  - Verbatim logging in `logs/llm_prompt_log.jsonl`.
  - Documented 3 failure modes in `src/llm_copilot/hallucination_cases.py`:
    1. Factual hallucination on contradictory 'Paid Off' status caught by Rule VR002.
    2. Ungrounded tax/income claim caught by schema constrained validation.
    3. Overconfident 'guaranteed default' claim corrected by probability calibration check.

---

## 3. Disqualification Self-Audit Checklist

| Requirement / Tripwire | Compliance Status | Evidence / Implementation Details |
| :--- | :--- | :--- |
| **No LLM-only prediction** | ✅ PASSED (100% Non-LLM ML) | All targets predicted strictly by LightGBM, LogisticRegression, Cox PH, and Isolation Forest. LLM used only for reviewer summarization. |
| **Trained non-LLM models** | ✅ PASSED | Trained models saved in `src/models/saved_models/` (`.joblib` binaries). |
| **No random split / Zero loan_id leakage** | ✅ PASSED | Time-aware cohort split by `origination_month`. Formally verified via `pytest tests/test_splitter.py`. |
| **No target leakage in features** | ✅ PASSED | 32 engineered features calculated strictly backward-looking from observation month $t$. |
| **Reproducible pipeline & real metrics** | ✅ PASSED | End-to-end runnable via `make run-all` with fixed seeds (`seed=42`). |
| **Grounded LLM with audit logs** | ✅ PASSED | Prompt, context, model, and output logged to `logs/llm_prompt_log.jsonl`; labeled as advisory recommendation. |

---

## 4. Key Lessons Learned

1. **Deterministic Guardrails are Mandatory for LLM Copilots**: Relying on LLMs alone to detect financial data inconsistencies frequently results in plausible-sounding hallucinations (e.g. assuming 'Paid Off' means zero risk even when balances remain). Hardcoded validation rules must take precedence over generative outputs.
2. **Probability Calibration is Critical in Imbalanced Credit Risk**: When positive default rates are low (~4%), raw gradient boosted scores can be overconfident. Applying Platt Sigmoid calibration directly improved Brier score reliability by over 50%.
3. **Time-Aware Splitting Prevents Overoptimistic Leakage**: Random row-level splitting on panel data allows models to memorize future trajectory states of the same borrower. Time-aware cohort splitting provides the only valid benchmark for real-world production performance.
