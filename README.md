# Loan Performance Intelligence Engine

> **Intain Campus FinTech Challenge 2026 — AI Track**

An end-to-end ML system for loan-level data profiling, performance prediction,
anomaly detection, scenario simulation, explainability, and grounded LLM-assisted review.

## Data Source

> **Using synthetic data generator** (`src/data_generation/generate.py`).
> No organizer-provided data was present in `data/raw/` at build time.
> To swap in organizer data: copy the organizer CSV files into `data/raw/` and
> re-run `make run-all` — the pipeline auto-detects their presence and skips generation.

## Setup

```bash
git clone https://github.com/ac265640/LoanScope.git
cd LoanScope
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY or GOOGLE_API_KEY
```

## Run Everything

```bash
make run-all
```

Or step by step:

```bash
make data          # generate synthetic data
make profile       # Task 1: profiling
make features      # feature engineering
make train         # Task 2: train prediction models
make survival      # Task 3: survival models
make anomaly       # Task 4: anomaly detection
make scenarios     # Task 5: stress simulation
make explain       # Task 6: explainability
make copilot       # Task 7: LLM copilot demo
make submission    # generate submission.csv
make test          # run pytest suite
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Loan Performance Intelligence Engine            │
├──────────┬────────────┬──────────────┬──────────────┬───────────┤
│  Data    │  Profiling │  Prediction  │  Survival    │  Anomaly  │
│  Gen     │  (Task 1)  │  (Task 2)    │  (Task 3)    │  (Task 4) │
│          │            │              │              │           │
│ 50K loans│ Missingness│ LightGBM     │ Kaplan-Meier │ IsoForest │
│ 36 months│ Outliers   │ + LogReg     │ Cox PH       │ + Rules   │
│ Messy    │ Drift      │ 5 targets    │ Transitions  │ SHAP exp  │
│ data     │ DQ score   │ Time-aware   │              │           │
├──────────┴────────────┴──────────────┴──────────────┴───────────┤
│  Scenarios (Task 5)  │  Explainability (Task 6)  │ LLM (Task 7)│
│  Base/Adverse/HighPP │  SHAP global/local        │ Grounded    │
│  Segment breakdowns  │  Model card, error anal.  │ Logged      │
│                      │  Uncertainty quantif.     │ Labeled     │
└──────────────────────┴───────────────────────────┴─────────────┘
```

## Results Summary

*(Populated after `make run-all`)*

| Model | Target | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| LightGBM | next_3m_delinquency | — | — | — |
| LightGBM | next_6m_delinquency | — | — | — |
| LightGBM | next_12m_default | — | — | — |
| LightGBM | next_12m_prepayment | — | — | — |
| LightGBM | next_state (macro-F1) | — | — | — |

## Deliverables

| File | Description |
|---|---|
| `submission/submission.csv` | Final scored output |
| `reports/data_intelligence_report.md` | Profiling results |
| `reports/model_card.md` | Model card |
| `reports/explainability_report.md` | Explainability |
| `reports/scenario_report.md` | Stress scenarios |
| `logs/llm_prompt_log.jsonl` | All LLM calls |
| `ai_development_log/AI_DEVELOPMENT_LOG.md` | Dev log |
