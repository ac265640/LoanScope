"""
Data Profiler Module
====================
Provides detailed column-level descriptive statistics for tabular numeric,
categorical, datetime, and identifier fields.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive column distributions and summary metadata.
    """
    profile: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        n_missing = int(series.isna().sum())
        pct_missing = float(n_missing / len(df) * 100)
        n_unique = int(series.nunique(dropna=True))

        col_stat: Dict[str, Any] = {
            "dtype": dtype_str,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 3),
            "n_unique": n_unique,
            "cardinality_ratio": round(n_unique / max(len(df), 1), 6),
        }

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            valid_vals = series.dropna()
            if len(valid_vals) > 0:
                col_stat.update({
                    "mean": float(round(valid_vals.mean(), 4)),
                    "std": float(round(valid_vals.std(), 4)) if len(valid_vals) > 1 else 0.0,
                    "min": float(round(valid_vals.min(), 4)),
                    "p25": float(round(valid_vals.quantile(0.25), 4)),
                    "median": float(round(valid_vals.median(), 4)),
                    "p75": float(round(valid_vals.quantile(0.75), 4)),
                    "max": float(round(valid_vals.max(), 4)),
                    "skewness": float(round(valid_vals.skew(), 4)) if len(valid_vals) > 2 else 0.0,
                    "zeros_count": int((valid_vals == 0).sum()),
                    "zeros_pct": float(round((valid_vals == 0).sum() / len(df) * 100, 2)),
                })
        else:
            val_counts = series.value_counts(dropna=False, normalize=True).head(10)
            top_values = {
                (str(k) if not pd.isna(k) else "<NULL>"): round(float(v) * 100, 2)
                for k, v in val_counts.items()
            }
            col_stat.update({
                "top_categories": top_values,
                "mode": str(series.mode()[0]) if not series.mode().empty else None,
            })

        profile["columns"][col] = col_stat

    return profile
