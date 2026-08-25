"""
Unified Command-Line Interface (CLI)
=====================================
Central CLI entrypoint for orchestrating all components of the
Loan Performance Intelligence Engine.

Usage:
  python src/pipeline/cli.py [command]

Available Commands:
  data         - Generate synthetic loan panel datasets (Phase 1)
  profile      - Run complete data intelligence and profiling suite (Task 1)
  train        - Train predictive ML models across all 5 targets (Task 2)
  survival     - Fit Kaplan-Meier, Cox PH, and Markov transition models (Task 3)
  anomaly      - Run Isolation Forest and exception prediction (Task 4)
  scenarios    - Run macro scenario stress simulations and Monte Carlo (Task 5)
  explain      - Generate TreeSHAP attributions and Model Card (Task 6)
  copilot      - Execute grounded LLM reviewer copilot demo (Task 7)
  submission   - Generate scored submission/submission.csv (Task 8)
  drift        - Run PSI & KS distributional drift monitoring
  fairness     - Conduct algorithmic fairness and disparate impact audit
  all          - Execute end-to-end pipeline from data to submission
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("LoanScope-CLI")


def main():
    parser = argparse.ArgumentParser(
        description="Loan Performance Intelligence Engine — Unified CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[
            "data", "profile", "train", "survival", "anomaly",
            "scenarios", "explain", "copilot", "submission",
            "drift", "fairness", "all",
        ],
        help="Pipeline command to execute",
    )

    args = parser.parse_args()
    cmd = args.command

    log.info(f"=== Executing LoanScope Pipeline Command: '{cmd.upper()}' ===")

    if cmd == "data":
        from src.data_generation.generate import main as run_data
        run_data()

    elif cmd == "profile":
        from src.profiling.run_profiling import main as run_profile
        run_profile()

    elif cmd == "train":
        from src.models.prediction.train_baseline import main as run_base
        from src.models.prediction.train_lgbm import main as run_lgbm
        from src.models.prediction.calibration import main as run_calib
        from src.models.prediction.evaluate import main as run_eval
        run_base()
        run_lgbm()
        run_calib()
        run_eval()

    elif cmd == "survival":
        from src.models.survival.kaplan_meier import main as run_km
        from src.models.survival.cox_ph import main as run_cox
        from src.models.survival.transition_matrix import main as run_tm
        from src.models.survival.competing_risks import main as run_cr
        from src.models.survival.evaluate_survival import main as run_surv_eval
        run_km()
        run_cox()
        run_tm()
        run_cr()
        run_surv_eval()

    elif cmd == "anomaly":
        from src.models.anomaly.isolation_forest import main as run_if
        from src.models.anomaly.exception_predictor import main as run_exc
        from src.models.anomaly.explain_anomalies import main as run_exp_anom
        run_if()
        run_exc()
        run_exp_anom()

    elif cmd == "scenarios":
        from src.scenarios.scenario_runner import main as run_scen
        from src.scenarios.monte_carlo import main as run_mc
        run_scen()
        run_mc()

    elif cmd == "explain":
        from src.explainability.global_importance import main as run_glob
        from src.explainability.local_explanation import main as run_loc
        from src.explainability.uncertainty import main as run_unc
        from src.explainability.error_analysis import main as run_err
        run_glob()
        run_loc()
        run_unc()
        run_err()

    elif cmd == "copilot":
        from src.llm_copilot.batch_copilot import run_batch_copilot
        run_batch_copilot()

    elif cmd == "submission":
        from src.pipeline.generate_submission import main as run_sub
        run_sub()

    elif cmd == "drift":
        from src.pipeline.drift_monitor import main as run_drift
        run_drift()

    elif cmd == "fairness":
        from src.explainability.fairness_audit import run_fairness_audit
        run_fairness_audit()

    elif cmd == "all":
        log.info("Starting complete end-to-end execution...")
        for sub_cmd in ["profile", "train", "survival", "anomaly", "scenarios", "explain", "copilot", "submission", "drift", "fairness"]:
            sys.argv = ["cli.py", sub_cmd]
            main()

    log.info(f"=== Pipeline Command '{cmd.upper()}' Completed Successfully ===")


if __name__ == "__main__":
    main()
