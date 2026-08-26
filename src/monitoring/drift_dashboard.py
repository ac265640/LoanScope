"""
Drift Monitoring Dashboard — Advanced Feature #3
=================================================
Interactive Streamlit dashboard showing per-feature drift (PSI / KS statistic)
between train and test splits, with pass/warn/fail thresholds visualized.

Run: streamlit run src/monitoring/drift_dashboard.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Core drift computation (standalone — works without Streamlit)
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
    train_file = RAW_DIR / "loan_monthly_performance_train.csv"
    test_file = RAW_DIR / "loan_monthly_performance_test.csv"

    train = pd.read_csv(train_file, low_memory=False)
    test = pd.read_csv(test_file, low_memory=False)

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


# ---------------------------------------------------------------------------
# Streamlit dashboard — only imported when running as streamlit app
# ---------------------------------------------------------------------------

def run_dashboard():
    """Launch the Streamlit drift monitoring dashboard."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Install streamlit and plotly: pip install streamlit plotly")
        return

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

    with st.spinner("Computing drift metrics..."):
        df = compute_drift_metrics()

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ PASS Features", int((df["status"] == "PASS").sum()))
    col2.metric("⚠️ WARN Features", int((df["status"] == "WARN").sum()))
    col3.metric("🚨 FAIL Features", int((df["status"] == "FAIL").sum()))

    # PSI bar chart
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
    st.plotly_chart(fig, use_container_width=True)

    # KS chart for numeric features
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
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.subheader("Full Drift Metrics Table")
    st.dataframe(
        df.style.applymap(
            lambda v: "background-color: #d4efdf" if v == "PASS"
            else "background-color: #fdebd0" if v == "WARN"
            else "background-color: #f5b7b1" if v == "FAIL" else "",
            subset=["status"],
        ),
        use_container_width=True,
    )

    st.caption(
        "Dashboard: `src/monitoring/drift_dashboard.py` | "
        "Run: `streamlit run src/monitoring/drift_dashboard.py`"
    )


if __name__ == "__main__":
    # When run directly (not via streamlit), compute and print metrics
    df = compute_drift_metrics()
    print(df.to_string(index=False))
    # Save static report
    out = REPO_ROOT / "reports" / "drift_monitoring_report.md"
    lines = [
        "# Feature Drift Monitoring Report\n",
        "## Train vs Test Distribution Drift (PSI / KS)\n",
        "| Feature | Type | PSI | KS | Status |",
        "|---------|------|-----|-----|--------|",
    ]
    for _, row in df.iterrows():
        ks_str = f"{row['KS']:.4f}" if row["KS"] is not None else "N/A"
        lines.append(f"| {row['feature']} | {row['type']} | {row['PSI']:.4f} | {ks_str} | {row['status']} |")
    lines += [
        "",
        "**Thresholds:** PSI < 0.10 = PASS, 0.10–0.25 = WARN, > 0.25 = FAIL",
        "",
        "**Launch dashboard:** `streamlit run src/monitoring/drift_dashboard.py`",
    ]
    out.write_text("\n".join(lines))
    print(f"\nReport written to {out}")
    # Try launching streamlit
    run_dashboard()
