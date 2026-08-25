# AI Development Log — Loan Performance Intelligence Engine

**Project**: Intain Campus FinTech Challenge 2026 — AI Track  
**AI Tool**: Google Antigravity (Claude Sonnet 4.6 Thinking)  
**Human Developer**: Amit  
**Log Start**: 2026-08-25

---

## How This Log Is Maintained

This file is updated **incrementally with each commit**, not retroactively.
Each entry records: what AI helped with, the representative prompt or instruction,
what was accepted, what was rejected and why, and the human review step.

---

## Entry 001 — 2026-08-25 | Project Scaffold

**AI Contribution**: Antigravity generated the full project directory structure,
`.gitignore`, `requirements.txt`, `Makefile`, and `README.md` skeleton from the
problem statement and instructions brief.

**Representative Prompt Given**:
> "go through instructions.txt and problem_statement.txt and start building"

**What Was Accepted**:
- Directory tree matching Section 2 of instructions exactly
- Makefile with one target per pipeline stage
- `.gitignore` covering Python, venv, data artifacts, checkpoints

**What Was Rejected / Modified**:
- Initial README had placeholder result table — kept as-is (will be filled after training)
- Reviewed `.gitignore` to ensure `data/raw/*.csv` is excluded (sensitive data) but `data/data_dictionary.md` is tracked

**Human Review**:
- Verified directory structure matches spec
- Confirmed API keys provided for LLM copilot

**Estimated AI Code Share This Phase**: ~95% (boilerplate scaffold)

---

## Entry 002 — 2026-08-25 | Synthetic Data Generator

*(To be updated after data generation commit)*

---

## Entry 003 — 2026-08-25 | Profiling Module

*(To be updated)*

---

## Disqualification Self-Audit (Final — To Be Completed)

- [ ] LLM-only prediction anywhere → ❌ not present
- [ ] No non-LLM trained model → ❌ LightGBM + LogReg trained
- [ ] Random splits with loan_id leakage → ❌ time-aware splitter with automated test
- [ ] Target leakage into features → ❌ features computed only from observation-month data
- [ ] No reproducible code → ❌ full Makefile + fixed seeds
- [ ] LLM narratives without grounding → ❌ all LLM calls grounded with context retrieval

---

## Lessons Learned

*(Updated throughout)*
