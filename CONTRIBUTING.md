# Contributing & Developer Navigation Guide

> **Intain Campus FinTech Challenge 2026 — AI Track**  
> **System**: Loan Performance Intelligence Engine

This document provides architectural boundaries, contribution standards, and developer guidelines to help reviewers and contributors navigate the codebase in under two minutes.

---

## 1. System Topology & Module Boundaries

The repository is organized into strict, decoupled modular subsystems:

```
src/
├── data_generation/      # Phase 1: Synthetic loan panel generator (50k loans × 36m)
├── profiling/            # Task 1: Column distributions, MNAR/MCAR tests, drift, DQ score
├── features/             # Non-leaking backward-looking rolling feature engineering
├── models/
│   ├── prediction/       # Task 2: Baseline LogReg & LightGBM + Platt calibration
│   ├── survival/         # Task 3: Kaplan-Meier, Cox PH, Markov transitions, Competing Risks
│   └── anomaly/          # Task 4: Isolation Forest & hybrid deterministic exception classifier
├── scenarios/            # Task 5: Macro shocks (Base/Adverse/High Prep) + Monte Carlo stress
├── explainability/       # Task 6: TreeSHAP global/local attributions, fairness audit, error analysis
├── llm_copilot/          # Task 7: Schema-grounded reviewer copilot, RAG, hallucination intercepts
└── pipeline/             # Task 8: Time-aware splitter, CLI, MLflow, drift monitor, submission generator
```

---

## 2. Core Non-Negotiable Invariants

Any contribution must strictly abide by these five architectural rules:

1. **Zero LLM Prediction**: All predictions (`next_3m_delinquency_flag`, `next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`, `anomaly_score`) must originate from trained, non-LLM models. LLMs are used solely for reviewer synthesis.
2. **Strict Time-Aware Cohort Splitting**: Partitioning must be performed by `origination_month` / `reporting_month`. No single `loan_id` may appear in both train and validation/test sets (`Intersection(Train_IDs, Val_IDs) == Ø`).
3. **No Target Leakage**: Engineered features computed at observation month $t$ must utilize only historical observations up to $t$.
4. **Deterministic Precedence**: Hardcoded validation rules (`validation_rules.json`) strictly override generative LLM narrative text.
5. **Grounded LLM Disclaimers**: Every generative output must include the advisory disclaimer: `Recommendation — not a decision.`

---

## 3. Running & Extending the Test Suite

Run the full automated test suite (30+ tests):
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run individual test suites
pytest tests/test_splitter.py -v           # Time-aware split & zero-leakage assertions
pytest tests/test_schema_validation.py -v   # Data invariants and column presence
pytest tests/test_features.py -v            # Non-leakage rolling calculations
pytest tests/test_submission.py -v          # Submission format and probability range bounds
pytest tests/test_validator.py -v           # Defensive input data validation
```

---

## 4. End-to-End Execution

Run the complete pipeline from scratch via bash script or Makefile:
```bash
# Option A: One-command bash script
./run_pipeline.sh

# Option B: Make runner
make run-all
```
