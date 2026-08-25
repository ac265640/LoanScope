"""
Correlation and Categorical Association Analysis
================================================
Calculates Pearson & Spearman correlations for continuous variables, and Cramér's V
for categorical-categorical associations to detect high dependency and multicollinearity.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramér's V statistic for categorical-categorical association.
    V ranges from 0.0 (no association) to 1.0 (perfect association).
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 10 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return 0.0

    confusion_matrix = pd.crosstab(valid["x"], valid["y"])
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = len(valid)
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Bias correction (Bergsma 2013)
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute numeric correlations and categorical association matrices.
    """
    results: Dict[str, Any] = {
        "numeric_pearson_high_pairs": [],
        "categorical_cramers_v_high_pairs": [],
        "multicollinearity_warnings": [],
    }

    # 1. Numeric Pearson & Spearman
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out pure binary flags from dense matrix for clarity
    dense_num_cols = [c for c in num_cols if df[c].nunique() > 2]

    if len(dense_num_cols) >= 2:
        corr_matrix = df[dense_num_cols].corr(method="pearson")
        for i in range(len(dense_num_cols)):
            for j in range(i + 1, len(dense_num_cols)):
                col1 = dense_num_cols[i]
                col2 = dense_num_cols[j]
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val) and abs(val) >= 0.4:
                    results["numeric_pearson_high_pairs"].append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "pearson_r": round(float(val), 4),
                        "strength": "High" if abs(val) >= 0.7 else "Moderate",
                    })
                    if abs(val) >= 0.85:
                        results["multicollinearity_warnings"].append(
                            f"Strong collinearity between `{col1}` and `{col2}` (r = {val:.3f}). Consider regularization or feature selection."
                        )

    # 2. Categorical Cramér's V
    cat_cols = [
        col for col in ["credit_score_band", "ltv_band", "dti_band", "state",
                        "loan_purpose", "occupancy_type", "property_type", "servicer_name", "document_status"]
        if col in df.columns
    ]

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            c1 = cat_cols[i]
            c2 = cat_cols[j]
            cv_val = cramers_v(df[c1], df[c2])
            if cv_val >= 0.15:
                results["categorical_cramers_v_high_pairs"].append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "cramers_v": round(cv_val, 4),
                    "association_strength": "High" if cv_val >= 0.35 else "Moderate",
                })

    return results
