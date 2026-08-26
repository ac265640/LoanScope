#!/usr/bin/env bash
# ==============================================================================
# Loan Performance Intelligence Engine — One-Command End-to-End Pipeline Runner
# ==============================================================================
# Executes the entire multi-task workflow from data generation/profiling to
# model training, survival analysis, anomaly detection, scenario simulation,
# explainability, grounded LLM copilot, and final submission validation.
#
# Usage:
#   ./run_pipeline.sh
# ==============================================================================

set -eo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
NC="\033[0m" # No Color

echo -e "${BOLD}${BLUE}==================================================================${NC}"
echo -e "${BOLD}${BLUE}   LOAN PERFORMANCE INTELLIGENCE ENGINE — END-TO-END PIPELINE    ${NC}"
echo -e "${BOLD}${BLUE}   Intain Campus FinTech Challenge 2026 — AI Track Submission    ${NC}"
echo -e "${BOLD}${BLUE}==================================================================${NC}"

# 1. Environment & Python Verification
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Activating virtual environment (venv)...${NC}"
    source venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR"

echo -e "\n${BOLD}[Step 1/10] Verifying Data & Generating Synthetic Datasets...${NC}"
python src/data_generation/generate.py

echo -e "\n${BOLD}[Step 2/10] Running Automated Leakage & Schema Validation Tests...${NC}"
pytest tests/ -v

echo -e "\n${BOLD}[Step 3/10] Running Task 1: Data Intelligence & Profiling...${NC}"
python src/profiling/run_profiling.py

echo -e "\n${BOLD}[Step 4/10] Running Task 2: Predictive Modeling & Calibration...${NC}"
python src/models/prediction/train_baseline.py
python src/models/prediction/train_lgbm.py
python src/models/prediction/calibration.py
python src/models/prediction/threshold_optimizer.py
python src/models/prediction/calibration_diagnostics.py
python src/models/prediction/evaluate.py

echo -e "\n${BOLD}[Step 5/10] Running Task 3: Survival, Cox PH & Markov Transitions...${NC}"
python src/models/survival/kaplan_meier.py
python src/models/survival/cox_ph.py
python src/models/survival/transition_matrix.py
python src/models/survival/competing_risks.py
python src/models/survival/evaluate_survival.py

echo -e "\n${BOLD}[Step 6/10] Running Task 4: Anomaly Detection & Exception Intelligence...${NC}"
python src/models/anomaly/isolation_forest.py
python src/models/anomaly/exception_predictor.py
python src/models/anomaly/explain_anomalies.py

echo -e "\n${BOLD}[Step 7/10] Running Task 5: Macro Scenarios & Monte Carlo Simulations...${NC}"
python src/scenarios/scenario_runner.py
python src/scenarios/monte_carlo.py

echo -e "\n${BOLD}[Step 8/10] Running Task 6: Explainability, Fairness Audit & Model Card...${NC}"
python src/explainability/global_importance.py
python src/explainability/local_explanation.py
python src/explainability/uncertainty.py
python src/explainability/fairness_audit.py
python src/explainability/error_analysis.py

echo -e "\n${BOLD}[Step 9/10] Running Task 7: Grounded LLM Copilot & Hallucination Audits...${NC}"
python src/llm_copilot/batch_copilot.py
python src/pipeline/drift_monitor.py

echo -e "\n${BOLD}[Step 10/10] Generating Final Scored submission.csv & Verifying Format...${NC}"
python src/pipeline/generate_submission.py
pytest tests/test_submission.py -v

echo -e "\n${BOLD}${GREEN}==================================================================${NC}"
echo -e "${BOLD}${GREEN}✅ ALL 10 PIPELINE STAGES COMPLETED AND VERIFIED SUCCESSFULLY!    ${NC}"
echo -e "${BOLD}${GREEN}==================================================================${NC}"
echo -e "Key Artifacts Generated:"
echo -e "  • Final Submission File   : submission/submission.csv"
echo -e "  • Data Intelligence Report: reports/data_intelligence_report.md"
echo -e "  • Model Card              : reports/model_card.md"
echo -e "  • Explainability Report   : reports/explainability_report.md"
echo -e "  • Scenario Stress Report  : reports/scenario_report.md"
echo -e "  • Reviewer Case Studies   : reports/anomaly_reviewer_cases.md"
echo -e "  • Calibration Report      : reports/calibration_report.md"
echo -e "  • Drift Monitoring Report : reports/drift_monitoring_report.md"
echo -e "  • LLM Prompt Audit Log    : logs/llm_prompt_log.jsonl"
echo -e "=================================================================="
