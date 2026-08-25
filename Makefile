.PHONY: all data profile features train survival anomaly scenarios explain copilot submission test run-all clean drift fairness monte-carlo competing-risks thresholds mlflow

PYTHON = PYTHONPATH=. python

run-all: data profile train survival anomaly scenarios explain copilot submission drift fairness
	@echo "=================================================================="
	@echo "✅ Full pipeline complete! All reports and submission.csv generated."
	@echo "=================================================================="

data:
	@echo "→ Generating synthetic data..."
	$(PYTHON) src/data_generation/generate.py

profile:
	@echo "→ Task 1: Data Intelligence & Profiling..."
	$(PYTHON) src/profiling/run_profiling.py

train:
	@echo "→ Task 2: Training prediction models..."
	$(PYTHON) src/models/prediction/train_baseline.py
	$(PYTHON) src/models/prediction/train_lgbm.py
	$(PYTHON) src/models/prediction/calibration.py
	$(PYTHON) src/models/prediction/evaluate.py

thresholds:
	@echo "→ Running classification threshold optimization..."
	$(PYTHON) src/models/prediction/threshold_optimizer.py

survival:
	@echo "→ Task 3: Survival modeling..."
	$(PYTHON) src/models/survival/kaplan_meier.py
	$(PYTHON) src/models/survival/cox_ph.py
	$(PYTHON) src/models/survival/transition_matrix.py
	$(PYTHON) src/models/survival/competing_risks.py
	$(PYTHON) src/models/survival/evaluate_survival.py

competing-risks:
	@echo "→ Fitting competing-risks CIF model..."
	$(PYTHON) src/models/survival/competing_risks.py

anomaly:
	@echo "→ Task 4: Anomaly & exception detection..."
	$(PYTHON) src/models/anomaly/isolation_forest.py
	$(PYTHON) src/models/anomaly/exception_predictor.py
	$(PYTHON) src/models/anomaly/explain_anomalies.py

scenarios:
	@echo "→ Task 5: Scenario & stress simulation..."
	$(PYTHON) src/scenarios/scenario_runner.py
	$(PYTHON) src/scenarios/monte_carlo.py

monte-carlo:
	@echo "→ Running Monte Carlo portfolio simulations..."
	$(PYTHON) src/scenarios/monte_carlo.py

explain:
	@echo "→ Task 6: Explainability..."
	$(PYTHON) src/explainability/global_importance.py
	$(PYTHON) src/explainability/local_explanation.py
	$(PYTHON) src/explainability/uncertainty.py
	$(PYTHON) src/explainability/error_analysis.py

fairness:
	@echo "→ Running Algorithmic Fairness & Disparate Impact audit..."
	$(PYTHON) src/explainability/fairness_audit.py

drift:
	@echo "→ Running Feature Drift Monitoring (PSI + KS)..."
	$(PYTHON) src/pipeline/drift_monitor.py

mlflow:
	@echo "→ Running MLflow experiment tracking pipeline..."
	$(PYTHON) src/pipeline/mlflow_tracking.py

copilot:
	@echo "→ Task 7: Grounded LLM Copilot demo..."
	$(PYTHON) src/llm_copilot/batch_copilot.py

submission:
	@echo "→ Generating submission.csv..."
	$(PYTHON) src/pipeline/generate_submission.py

test:
	@echo "→ Running comprehensive pytest suite..."
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
