"""
Agentic Experiment Runner — Advanced Feature #8
===============================================
Orchestrator that autonomously sweeps model types × feature sets from a
YAML config, compares results, logs to MLflow, and proposes the next
configuration to try based on prior results (best-first heuristic).

Run: PYTHONPATH=. python src/pipeline/experiment_runner.py --sweep configs/sweep.yaml
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Try to import MLflow for tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _load_sweep_config(config_path: Path) -> Dict:
    """Load YAML sweep config (uses json as fallback if yaml not installed)."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Read as JSON fallback
        with open(config_path) as f:
            return json.load(f)


def _load_data(target: str = "next_12m_default_flag"):
    """Load features and target from raw data, using top numeric columns."""
    df = pd.read_csv(RAW_DIR / "loan_monthly_performance_train.csv", low_memory=False)
    feature_cols = [
        "loan_age_months", "remaining_term_months", "original_balance",
        "current_balance", "interest_rate", "days_past_due",
        "modification_flag", "prepayment_flag", "default_flag",
    ]
    available = [c for c in feature_cols if c in df.columns]
    target_col = target if target in df.columns else "default_flag"

    df_clean = df[available + [target_col]].dropna()
    X = df_clean[available].values
    y = df_clean[target_col].values.astype(float)
    return X, y, available


def _get_feature_set(X: np.ndarray, feature_names: List[str], feature_set: str) -> np.ndarray:
    """Return subset of features based on feature set name."""
    if feature_set == "all":
        return X
    elif feature_set == "credit":
        idx = [i for i, n in enumerate(feature_names) if any(k in n for k in ["interest", "balance", "score"])]
        return X[:, idx] if idx else X
    elif feature_set == "behavioral":
        idx = [i for i, n in enumerate(feature_names) if any(k in n for k in ["days", "flag", "age"])]
        return X[:, idx] if idx else X
    return X


def _train_evaluate(model_type: str, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    params: Dict) -> Dict:
    """Train a model and return evaluation metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=500,
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 5),
            random_state=42, n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 4),
                random_state=42, verbose=-1,
            )
        except ImportError:
            model = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 50),
                random_state=42,
            )
    else:
        model = LogisticRegression(max_iter=200, random_state=42)

    model.fit(X_tr_s, y_tr)
    y_prob = model.predict_proba(X_val_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
    f1 = f1_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)

    return {"auc": round(auc, 4), "f1": round(f1, 4),
            "precision": round(precision, 4), "recall": round(recall, 4)}


def _propose_next_config(results: List[Dict]) -> Dict:
    """
    Agentic heuristic: propose next config to try based on prior results.
    Strategy: select best-performing model_type, then try a more complex feature set.
    """
    if not results:
        return {}
    best = max(results, key=lambda r: r["metrics"]["auc"])
    suggestion = {
        "model_type": best["model_type"],
        "feature_set": "all",
        "params": {k: v * 1.5 if isinstance(v, float) else v + 10
                   for k, v in best["params"].items()},
        "rationale": (
            f"Best so far: {best['model_type']} with AUC={best['metrics']['auc']:.4f}. "
            f"Proposing increased capacity on same architecture with all features."
        ),
    }
    return suggestion


def run_sweep(config_path: Path):
    print(f"[experiment_runner] Loading sweep config from {config_path}")
    config = _load_sweep_config(config_path)

    X, y, feature_names = _load_data(target=config.get("target", "next_12m_default_flag"))

    # Time-based train/val split (80/20)
    split_idx = int(0.8 * len(X))
    X_tr, X_val = X[:split_idx], X[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    results = []
    exp_id = 0

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.get("experiment_name", "agentic_sweep"))

    for run_cfg in config.get("runs", []):
        model_type = run_cfg["model_type"]
        feature_set = run_cfg.get("feature_set", "all")
        params = run_cfg.get("params", {})

        X_tr_f = _get_feature_set(X_tr, feature_names, feature_set)
        X_val_f = _get_feature_set(X_val, feature_names, feature_set)

        print(f"  Experiment {exp_id}: {model_type} | features={feature_set} | params={params}")

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"{model_type}_{feature_set}_{exp_id}"):
                mlflow.log_params({"model_type": model_type, "feature_set": feature_set, **params})
                metrics = _train_evaluate(model_type, X_tr_f, y_tr, X_val_f, y_val, params)
                mlflow.log_metrics(metrics)
        else:
            metrics = _train_evaluate(model_type, X_tr_f, y_tr, X_val_f, y_val, params)

        result = {
            "exp_id": exp_id,
            "model_type": model_type,
            "feature_set": feature_set,
            "params": params,
            "metrics": metrics,
        }
        results.append(result)
        print(f"    → AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")
        exp_id += 1

    # Agentic proposal
    next_cfg = _propose_next_config(results)

    # Save results
    out_path = LOGS_DIR / "sweep_results.json"
    summary = {"runs": results, "next_proposed_config": next_cfg}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[experiment_runner] Completed {len(results)} experiments.")
    print(f"[experiment_runner] Results saved to {out_path}")
    if next_cfg:
        print(f"[experiment_runner] AGENTIC PROPOSAL: {next_cfg['rationale']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Agentic Experiment Runner")
    parser.add_argument("--sweep", type=str, default="configs/sweep.yaml",
                        help="Path to sweep config YAML/JSON")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.sweep
    if not config_path.exists():
        print(f"Config not found at {config_path}, using default inline config")
        # Use default config
        config_path = None

    if config_path is None:
        # Inline default sweep
        config = {
            "experiment_name": "agentic_sweep_default",
            "target": "default_flag",
            "runs": [
                {"model_type": "logistic_regression", "feature_set": "all", "params": {"C": 1.0}},
                {"model_type": "logistic_regression", "feature_set": "behavioral", "params": {"C": 0.1}},
                {"model_type": "random_forest", "feature_set": "all", "params": {"n_estimators": 50, "max_depth": 5}},
                {"model_type": "gradient_boosting", "feature_set": "all", "params": {"n_estimators": 100, "learning_rate": 0.05}},
                {"model_type": "gradient_boosting", "feature_set": "credit", "params": {"n_estimators": 50, "learning_rate": 0.1}},
            ],
        }
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = Path(f.name)

    run_sweep(config_path)


if __name__ == "__main__":
    main()
