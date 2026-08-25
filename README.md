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
make data          # Phase 1: Generate synthetic 50k loans panel data
make test          # Run automated zero-leakage pytest suite
make profile       # Task 1: Generate Data Intelligence report
make train         # Task 2: Train baseline & LightGBM prediction models
make survival      # Task 3: Fit Kaplan-Meier, Cox PH & Markov transition matrix
make anomaly       # Task 4: Run Isolation Forest & generate 25 reviewer cases
make scenarios     # Task 5: Run Base, Adverse Credit & High Prepayment stress
make explain       # Task 6: Compute SHAP values, error audit, and Model Card
make copilot       # Task 7: Execute grounded LLM reviewer copilot demo
make submission    # Generate submission/submission.csv
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
| **Final Submission File** | [`submission/submission.csv`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/submission/submission.csv) | Final scored test dataset with probabilities, anomaly scores, top drivers, and actions. |
| **Data Intelligence Report** | [`reports/data_intelligence_report.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/reports/data_intelligence_report.md) | Full distributions, missingness (MNAR/MCAR), outliers, drift, and DQ scores. |
| **Model Card** | [`reports/model_card.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/reports/model_card.md) | Official model specifications, data lineage, validation method, and limitations. |
| **Explainability Report** | [`reports/explainability_report.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/reports/explainability_report.md) | Global SHAP importance, local waterfalls, uncertainty, and FP/FN audits. |
| **Scenario Stress Report** | [`reports/scenario_report.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/reports/scenario_report.md) | Macro simulations (Base, Adverse Credit, High Prepayment) and segment curves. |
| **Anomaly Reviewer Cases** | [`reports/anomaly_reviewer_cases.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/reports/anomaly_reviewer_cases.md) | 25 detailed reviewer cases with plain-English diagnostic explanations. |
| **LLM Prompt Log** | [`logs/llm_prompt_log.jsonl`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/logs/llm_prompt_log.jsonl) | Verbatim audit log of all grounded copilot prompt payloads and responses. |
| **AI Development Log** | [`ai_development_log/AI_DEVELOPMENT_LOG.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/ai_development_log/AI_DEVELOPMENT_LOG.md) | Incremental dev notes, prompts, accepted/rejected outputs, and self-audit. |
| **Walkthrough Notebook** | [`notebooks/end_to_end_walkthrough.ipynb`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/notebooks/end_to_end_walkthrough.ipynb) | End-to-end reproducible walkthrough notebook across all 8 tasks. |
| **Demo Video Outline** | [`demo_script.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/demo_script.md) | Exact 5-minute video presentation script matching Section 14 flow. |

---

## 5. Disqualification Self-Audit Status

- [x] **No LLM-only prediction** (All predictions generated by non-LLM models: LightGBM, LogisticRegression, Cox PH, Isolation Forest).
- [x] **Trained non-LLM models saved** (`.joblib` binaries in `src/models/saved_models/`).
- [x] **Time-aware split with zero `loan_id` leakage** (Asserted via `pytest tests/test_splitter.py`).
- [x] **Zero target leakage into features** (32 features calculated strictly from backward-looking historical observations).
- [x] **Reproducible code and evaluation metrics** (Full `Makefile` with fixed seed `42`).
- [x] **Grounded LLM reviewer summaries** (Grounded retrieval, verbatim prompt logs, labeled as advisory recommendation).
