"""
Drift Monitoring Dashboard — Advanced Feature #3
=================================================
Interactive Streamlit dashboard showing per-feature drift (PSI / KS statistic)
between train and test splits, with pass/warn/fail thresholds visualized.

Run locally (full scale data):
    streamlit run src/monitoring/drift_dashboard.py

Streamlit Community Cloud deployment:
    Main file path: src/monitoring/drift_dashboard.py
    Requirements:   requirements-dashboard.txt
    Secrets:        None required

The dashboard self-generates a representative sample dataset on first load
if data/raw/ is empty (cloud-safe — no manual setup required for visitors).
See DEPLOYMENT.md at repo root for full step-by-step deployment instructions.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution — always relative to this file, never hardcoded/absolute
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DIR / "loan_monthly_performance_train.csv"
TEST_FILE = RAW_DIR / "loan_monthly_performance_test.csv"

# ---------------------------------------------------------------------------
# Lite / demo mode settings
# ---------------------------------------------------------------------------
# On Streamlit Community Cloud (free tier, ~1 GB RAM) generating the full
# 50,000-loan × 36-month dataset (~874K rows) is too memory-intensive.
# Instead, we generate a 7,000-loan × 24-month sample — still large enough
# for real, meaningful drift statistics while staying well within free-tier limits.
#
# Force full scale locally by setting DASHBOARD_LITE=0, or force lite mode
# with DASHBOARD_LITE=1 for testing.
_LITE_ENV = os.environ.get("DASHBOARD_LITE", "").strip().lower()
LITE_MODE = _LITE_ENV in ("1", "true", "yes")
LITE_N_LOANS = 3_000
LITE_MAX_MONTHS = 18

# ---------------------------------------------------------------------------
# Core drift computation (standalone — no model artifacts required)
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "days_past_due",
]
CATEGORICAL_COLS = [
    "credit_score_band", "ltv_band", "dti_band", "state",
    "loan_purpose", "occupancy_type", "current_status",
]


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    combined = np.concatenate([expected, actual])
    breakpoints = np.nanpercentile(combined, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)

    e_pct = np.where(e_counts == 0, 1e-6, e_counts / len(expected))
    a_pct = np.where(a_counts == 0, 1e-6, a_counts / len(actual))

    return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """KS two-sample statistic."""
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(a, b)
    return float(stat)


def _categorical_psi(train_series: pd.Series, test_series: pd.Series) -> float:
    cats = pd.Categorical(
        pd.concat([train_series, test_series]).astype(str)
    ).categories
    e = train_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=1e-6)
    a = test_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=1e-6)
    return float(np.sum((e - a) * np.log(e / a)))


def compute_drift_metrics() -> pd.DataFrame:
    """Compute PSI and KS for all features between train and test."""
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)

    rows = []
    for col in NUMERIC_COLS:
        if col not in train.columns:
            continue
        t = train[col].dropna().values
        v = test[col].dropna().values
        psi = _psi(t, v) if len(t) > 10 and len(v) > 10 else 0.0
        ks = _ks_stat(t, v) if len(t) > 10 and len(v) > 10 else 0.0
        status = "PASS" if psi < 0.1 else ("WARN" if psi < 0.25 else "FAIL")
        rows.append({"feature": col, "type": "numeric", "PSI": round(psi, 4),
                     "KS": round(ks, 4), "status": status})

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        psi = _categorical_psi(train[col].fillna("MISSING"), test[col].fillna("MISSING"))
        status = "PASS" if psi < 0.1 else ("WARN" if psi < 0.25 else "FAIL")
        rows.append({"feature": col, "type": "categorical", "PSI": round(psi, 4),
                     "KS": None, "status": status})

    return pd.DataFrame(rows)


def _generate_sample_data(n_loans: int, max_months: int) -> None:
    """
    Invoke the repo's data generator to produce train/test CSVs.
    Uses importlib so the import works regardless of Python path / working dir.
    """
    gen_path = REPO_ROOT / "src" / "data_generation" / "generate.py"
    spec = importlib.util.spec_from_file_location("_dashboard_generate", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    sys.modules["_dashboard_generate"] = gen_mod
    spec.loader.exec_module(gen_mod)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gen_mod.main(n_loans=n_loans, max_months=max_months, skip_targets=True)


# ---------------------------------------------------------------------------
# Streamlit cached helpers (defined at module level for stable cache keys)
# ---------------------------------------------------------------------------

def _ensure_data_resource() -> bool:
    """Return True if data had to be generated (demo mode), False otherwise."""
    if TRAIN_FILE.exists() and TEST_FILE.exists() and not LITE_MODE:
        return False
    if TRAIN_FILE.exists() and TEST_FILE.exists() and LITE_MODE:
        return True
    _generate_sample_data(n_loans=LITE_N_LOANS, max_months=LITE_MAX_MONTHS)
    return True


try:
    import streamlit as st
    import plotly.express as px
    _ensure_data_cached = st.cache_resource(show_spinner=False)(_ensure_data_resource)
    _compute_drift_cached = st.cache_data(show_spinner=False)(compute_drift_metrics)
except Exception:
    _ensure_data_cached = _ensure_data_resource
    _compute_drift_cached = compute_drift_metrics


# ---------------------------------------------------------------------------
# Streamlit dashboard UI
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    """Render the Streamlit drift monitoring dashboard."""
    st.set_page_config(
        page_title="LoanScope — Drift Monitoring Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 LoanScope — Feature Drift Monitoring Dashboard")
    st.markdown(
        "**PSI thresholds:** 🟢 PASS (< 0.10) &nbsp;|&nbsp; 🟡 WARN (0.10–0.25) "
        "&nbsp;|&nbsp; 🔴 FAIL (> 0.25) &nbsp;&nbsp; — "
        "Comparing **train** vs **test** feature distributions."
    )

    # ------------------------------------------------------------------
    # Step 1: Ensure data exists — self-generate on first cloud load
    # ------------------------------------------------------------------
    data_was_missing = not (TRAIN_FILE.exists() and TEST_FILE.exists())

    if data_was_missing:
        with st.spinner(
            f"⏳ **First load** — generating sample dataset "
            f"({LITE_N_LOANS:,} loans, {LITE_MAX_MONTHS} months). "
            "This takes ~2–5 s and only happens once per deployment…"
        ):
            is_demo = _ensure_data_cached()
    else:
        is_demo = _ensure_data_cached()

    # ------------------------------------------------------------------
    # Step 2: Show demo-mode banner (honest framing)
    # ------------------------------------------------------------------
    if is_demo:
        st.info(
            "🔬 **Hosted Demo Mode** — This deployment uses a reduced-scale sample "
            f"(**{LITE_N_LOANS:,} loans × {LITE_MAX_MONTHS} months**) for free-tier "
            "performance reasons. Drift statistics and charts are real and representative. "
            "For the full-scale run (50,000 loans × 36 months), clone the repo and run "
            "`make run-all` locally.",
            icon="ℹ️",
        )

    # ------------------------------------------------------------------
    # Step 3: Compute drift metrics (cached)
    # ------------------------------------------------------------------
    with st.spinner("📐 Computing drift metrics…"):
        df = _compute_drift_cached()

    # ------------------------------------------------------------------
    # KPI summary row
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ PASS Features", int((df["status"] == "PASS").sum()))
    col2.metric("⚠️ WARN Features", int((df["status"] == "WARN").sum()))
    col3.metric("🚨 FAIL Features", int((df["status"] == "FAIL").sum()))

    # ------------------------------------------------------------------
    # PSI bar chart
    # ------------------------------------------------------------------
    color_map = {"PASS": "#27ae60", "WARN": "#f39c12", "FAIL": "#e74c3c"}
    df_sorted = df.sort_values("PSI", ascending=False)
    fig = px.bar(
        df_sorted, x="feature", y="PSI", color="status",
        color_discrete_map=color_map,
        title="PSI by Feature (higher = more drift)",
        labels={"PSI": "Population Stability Index", "feature": "Feature"},
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange",
                  annotation_text="WARN threshold (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="FAIL threshold (0.25)")
    fig.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    # KS chart for numeric features
    # ------------------------------------------------------------------
    numeric_df = df[df["type"] == "numeric"].dropna(subset=["KS"])
    if len(numeric_df) > 0:
        fig2 = px.bar(
            numeric_df.sort_values("KS", ascending=False),
            x="feature", y="KS", color="status",
            color_discrete_map=color_map,
            title="KS Statistic by Numeric Feature",
        )
        fig2.add_hline(y=0.05, line_dash="dash", line_color="orange",
                       annotation_text="p<0.05 threshold")
        st.plotly_chart(fig2, width="stretch")

    # ------------------------------------------------------------------
    # Full metrics table — use .map() (pandas ≥ 2.1, replaces applymap)
    # ------------------------------------------------------------------
    st.subheader("Full Drift Metrics Table")

    def _style_status(val: object) -> str:
        if val == "PASS":
            return "background-color: #d4efdf"
        elif val == "WARN":
            return "background-color: #fdebd0"
        elif val == "FAIL":
            return "background-color: #f5b7b1"
        return ""

    st.dataframe(
        df.style.map(_style_status, subset=["status"]),
        width="stretch",
    )

    st.caption(
        "Dashboard: `src/monitoring/drift_dashboard.py` | "
        "Run locally: `streamlit run src/monitoring/drift_dashboard.py` | "
        "Deploy: see `DEPLOYMENT.md`"
    )


# ---------------------------------------------------------------------------
# Entry point: Detect whether invoked via Streamlit or pure Python CLI
# ---------------------------------------------------------------------------

def _is_running_under_streamlit() -> bool:
    """Return True if executing inside an active Streamlit server session."""
    try:
        import streamlit.runtime
        return streamlit.runtime.exists()
    except Exception:
        return False


if _is_running_under_streamlit():
    # Streamlit Cloud execution: Load and render the full multi-page showcase
    import importlib.util
    app_path = Path(__file__).resolve().parent / "app.py"
    if app_path.exists():
        spec = importlib.util.spec_from_file_location("showcase_app", app_path)
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
    else:
        run_dashboard()
elif __name__ == "__main__":
    # CLI execution (`python src/monitoring/drift_dashboard.py`)
    if not (TRAIN_FILE.exists() and TEST_FILE.exists()):
        print("Data files not found. Auto-generating lite dataset...")
        _generate_sample_data(n_loans=LITE_N_LOANS, max_months=LITE_MAX_MONTHS)

    df = compute_drift_metrics()
    print(df.to_string(index=False))

    out = REPO_ROOT / "reports" / "drift_monitoring_report.md"
    lines = [
        "# Feature Drift Monitoring Report\n",
        "## Train vs Test Distribution Drift (PSI / KS)\n",
        "| Feature | Type | PSI | KS | Status |",
        "|---------|------|-----|-----|--------|",
    ]
    for _, row in df.iterrows():
        ks_str = f"{row['KS']:.4f}" if row["KS"] is not None else "N/A"
        lines.append(
            f"| {row['feature']} | {row['type']} | {row['PSI']:.4f} | {ks_str} | {row['status']} |"
        )
    lines += [
        "",
        "**Thresholds:** PSI < 0.10 = PASS, 0.10–0.25 = WARN, > 0.25 = FAIL",
        "",
        "**Launch dashboard:** `streamlit run src/monitoring/drift_dashboard.py`",
    ]
    out.write_text("\n".join(lines))
    print(f"\nReport written to {out}")

