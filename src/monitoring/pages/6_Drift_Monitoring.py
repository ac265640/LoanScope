"""
Page 6: Feature Drift Monitoring Dashboard
===========================================
Interactive feature drift surveillance showing Population Stability Index (PSI)
and Kolmogorov-Smirnov (KS) statistics between train and test distributions.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Drift Monitoring | LoanScope",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Path resolution — always relative to this file
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DIR / "loan_monthly_performance_train.csv"
TEST_FILE = RAW_DIR / "loan_monthly_performance_test.csv"

_LITE_ENV = os.environ.get("DASHBOARD_LITE", "").strip().lower()
LITE_MODE = _LITE_ENV in ("1", "true", "yes")
LITE_N_LOANS = 3_000
LITE_MAX_MONTHS = 18

NUMERIC_COLS = [
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "days_past_due",
]
CATEGORICAL_COLS = [
    "credit_score_band", "ltv_band", "dti_band", "state",
    "loan_purpose", "occupancy_type", "current_status",
]

# Sidebar
with st.sidebar:
    st.markdown("### Drift Monitoring")
    st.caption("PSI & KS Distribution Stability")
    st.info(
        "**Hosted Demo Mode**\n\n"
        f"Comparing train vs test distributions ({LITE_N_LOANS:,} loans sample)."
    )

# ---------------------------------------------------------------------------
# Core drift computation
# ---------------------------------------------------------------------------

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
    """Invoke generator for train/test CSVs with skip_targets=True."""
    gen_path = REPO_ROOT / "src" / "data_generation" / "generate.py"
    spec = importlib.util.spec_from_file_location("_dashboard_generate", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    sys.modules["_dashboard_generate"] = gen_mod
    spec.loader.exec_module(gen_mod)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gen_mod.main(n_loans=n_loans, max_months=max_months, skip_targets=True)


# Module-level caching for stable performance
@st.cache_resource(show_spinner=False)
def _ensure_data_resource() -> bool:
    if TRAIN_FILE.exists() and TEST_FILE.exists() and not LITE_MODE:
        return False
    if TRAIN_FILE.exists() and TEST_FILE.exists() and LITE_MODE:
        return True
    _generate_sample_data(n_loans=LITE_N_LOANS, max_months=LITE_MAX_MONTHS)
    return True


@st.cache_data(show_spinner=False)
def _compute_drift_cached() -> pd.DataFrame:
    return compute_drift_metrics()


# Main Page Body
st.title("Feature Drift Surveillance Dashboard")
st.markdown(
    "**PSI Thresholds:** PASS (< 0.10) | WARN (0.10–0.25) | FAIL (> 0.25) — "
    "Comparing **historical train** vs **out-of-time test** distributions."
)

data_was_missing = not (TRAIN_FILE.exists() and TEST_FILE.exists())
if data_was_missing:
    with st.spinner("Generating sample dataset..."):
        is_demo = _ensure_data_resource()
else:
    is_demo = _ensure_data_resource()

if is_demo:
    st.info(
        "**Hosted Demo Mode** — Reduced-scale sample "
        f"({LITE_N_LOANS:,} loans × {LITE_MAX_MONTHS} months). "
        "Drift metrics are real and representative. Full run available via `make run-all`."
    )

with st.spinner("Computing drift metrics..."):
    df = _compute_drift_cached()

# Summary KPIs
col1, col2, col3 = st.columns(3)
col1.metric("PASS Features", int((df["status"] == "PASS").sum()))
col2.metric("WARN Features", int((df["status"] == "WARN").sum()))
col3.metric("FAIL Features", int((df["status"] == "FAIL").sum()))

# PSI bar chart
color_map = {"PASS": "#27ae60", "WARN": "#f39c12", "FAIL": "#e74c3c"}
df_sorted = df.sort_values("PSI", ascending=False)
fig = px.bar(
    df_sorted, x="feature", y="PSI", color="status",
    color_discrete_map=color_map,
    title="Population Stability Index (PSI) by Feature",
    labels={"PSI": "PSI Score", "feature": "Feature"},
)
fig.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="WARN threshold (0.10)")
fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="FAIL threshold (0.25)")
fig.update_layout(xaxis_tickangle=-45, height=430)
st.plotly_chart(fig, use_container_width=True)

# KS chart for numeric features
numeric_df = df[df["type"] == "numeric"].dropna(subset=["KS"])
if len(numeric_df) > 0:
    fig2 = px.bar(
        numeric_df.sort_values("KS", ascending=False),
        x="feature", y="KS", color="status",
        color_discrete_map=color_map,
        title="Kolmogorov-Smirnov (KS) Statistic by Numeric Feature",
    )
    fig2.add_hline(y=0.05, line_dash="dash", line_color="orange", annotation_text="p<0.05 threshold")
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

# Full table
st.subheader("Full Drift Metrics Table")

def _style_status(val: object) -> str:
    if val == "PASS":
        return "background-color: #d4efdf; color: #1e293b"
    elif val == "WARN":
        return "background-color: #fdebd0; color: #1e293b"
    elif val == "FAIL":
        return "background-color: #f5b7b1; color: #1e293b"
    return ""

st.dataframe(
    df.style.map(_style_status, subset=["status"]),
    use_container_width=True,
)
