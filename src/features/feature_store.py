"""
Feature-Store Pipeline — Advanced Feature #9
============================================
Versioned, reusable feature-store pattern: features computed once, written
to data/processed/feature_store/ with schema/version manifest, and reused
by all downstream models.

Run: PYTHONPATH=. python src/features/feature_store.py
"""

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = REPO_ROOT / "data"
STORE_DIR = REPO_ROOT / "data" / "processed" / "feature_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_VERSION = "v1.0.0"

# ---------------------------------------------------------------------------
# Feature Registry
# ---------------------------------------------------------------------------

FEATURE_REGISTRY = {
    "loan_age_months": {
        "description": "Number of months since loan origination",
        "source_columns": ["loan_age_months"],
        "computation": "direct copy",
        "dtype": "float",
        "category": "temporal",
    },
    "remaining_term_months": {
        "description": "Remaining months on loan term",
        "source_columns": ["remaining_term_months"],
        "computation": "direct copy",
        "dtype": "float",
        "category": "temporal",
    },
    "current_balance_ratio": {
        "description": "Current balance as fraction of original balance",
        "source_columns": ["current_balance", "original_balance"],
        "computation": "current_balance / original_balance",
        "dtype": "float",
        "category": "credit",
    },
    "interest_rate": {
        "description": "Current interest rate on loan",
        "source_columns": ["interest_rate"],
        "computation": "direct copy",
        "dtype": "float",
        "category": "rate",
    },
    "days_past_due_capped": {
        "description": "Days past due, capped at 90 to reduce outlier impact",
        "source_columns": ["days_past_due"],
        "computation": "min(days_past_due, 90)",
        "dtype": "float",
        "category": "delinquency",
    },
    "is_delinquent": {
        "description": "Binary flag: 1 if any days past due > 0",
        "source_columns": ["days_past_due"],
        "computation": "(days_past_due > 0).astype(int)",
        "dtype": "int",
        "category": "delinquency",
    },
    "modification_flag": {
        "description": "Binary: loan has been modified",
        "source_columns": ["modification_flag"],
        "computation": "direct copy",
        "dtype": "int",
        "category": "credit",
    },
    "credit_score_band_encoded": {
        "description": "Credit score band encoded as ordinal integer",
        "source_columns": ["credit_score_band"],
        "computation": "ordinal encoding: <620=0, 620-659=1, 660-699=2, 700-739=3, 740-779=4, 780+=5",
        "dtype": "int",
        "category": "credit",
    },
    "ltv_band_encoded": {
        "description": "LTV band encoded as ordinal integer",
        "source_columns": ["ltv_band"],
        "computation": "ordinal encoding: low to high",
        "dtype": "int",
        "category": "credit",
    },
    "dti_band_encoded": {
        "description": "DTI band encoded as ordinal integer",
        "source_columns": ["dti_band"],
        "computation": "ordinal encoding: low to high",
        "dtype": "int",
        "category": "credit",
    },
    "loan_purpose_encoded": {
        "description": "Loan purpose encoded as integer category",
        "source_columns": ["loan_purpose"],
        "computation": "label encoding",
        "dtype": "int",
        "category": "loan_attributes",
    },
    "occupancy_type_encoded": {
        "description": "Occupancy type encoded as integer category",
        "source_columns": ["occupancy_type"],
        "computation": "label encoding",
        "dtype": "int",
        "category": "loan_attributes",
    },
}

BAND_ORDINALS = {
    "credit_score_band": {
        "<620": 0, "620-659": 1, "620-699": 1, "660-699": 2, "660-719": 2,
        "700-739": 3, "700-759": 3, "740-779": 4, "760-779": 4, "780+": 5,
    },
    "ltv_band": {
        "<60%": 0, "60-70%": 1, "70-80%": 2, "80-90%": 3, "90-95%": 4, "95%+": 5,
        "<=60%": 0, "60%-70%": 1, "70%-80%": 2,
    },
    "dti_band": {
        "<20%": 0, "20-30%": 1, "20%-30%": 1, "30-40%": 2, "30%-40%": 2,
        "40-50%": 3, "40%-50%": 3, "50%+": 4,
    },
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _encode_band(series: pd.Series, band_col: str) -> pd.Series:
    mapping = BAND_ORDINALS.get(band_col, {})
    encoded = series.map(mapping)
    # For unknown values, assign median
    median_val = int(np.nanmedian([v for v in mapping.values()]))
    return encoded.fillna(median_val).astype(int)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all registered features from raw data."""
    features = pd.DataFrame(index=df.index)
    features["loan_id"] = df["loan_id"]
    features["month_index"] = df.get("month_index", pd.Series(range(len(df)), index=df.index))

    # Temporal
    features["loan_age_months"] = df.get("loan_age_months", pd.Series(0, index=df.index)).fillna(0)
    features["remaining_term_months"] = df.get("remaining_term_months", pd.Series(0, index=df.index)).fillna(0)

    # Credit ratio
    orig_bal = df.get("original_balance", pd.Series(1, index=df.index)).replace(0, 1)
    curr_bal = df.get("current_balance", orig_bal)
    features["current_balance_ratio"] = (curr_bal / orig_bal).clip(0, 2).fillna(1.0)
    features["interest_rate"] = df.get("interest_rate", pd.Series(0, index=df.index)).fillna(0)

    # Delinquency
    dpd = df.get("days_past_due", pd.Series(0, index=df.index)).fillna(0)
    features["days_past_due_capped"] = dpd.clip(0, 90)
    features["is_delinquent"] = (dpd > 0).astype(int)
    features["modification_flag"] = df.get("modification_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Encoded categoricals
    for col, encoded_col, band_col in [
        ("credit_score_band", "credit_score_band_encoded", "credit_score_band"),
        ("ltv_band", "ltv_band_encoded", "ltv_band"),
        ("dti_band", "dti_band_encoded", "dti_band"),
    ]:
        if col in df.columns:
            features[encoded_col] = _encode_band(df[col].astype(str), band_col)
        else:
            features[encoded_col] = 0

    # Label encode loan_purpose, occupancy_type
    for col in ["loan_purpose", "occupancy_type"]:
        enc_col = col + "_encoded"
        if col in df.columns:
            cats = df[col].astype(str).astype("category")
            features[enc_col] = cats.cat.codes
        else:
            features[enc_col] = 0

    return features


def _compute_fingerprint(df: pd.DataFrame) -> str:
    """Compute a lightweight fingerprint of input data for versioning."""
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update(str(sorted(df.columns.tolist())).encode())
    if len(df) > 0:
        h.update(str(df.iloc[0].to_dict()).encode())
    return h.hexdigest()[:12]


def build_feature_store(split: str = "train") -> Path:
    """
    Build and save the feature store for the given split.
    Returns path to saved feature parquet.
    """
    print(f"[feature_store] Building feature store for split={split}...")

    # Load raw data
    fname = f"loan_monthly_performance_{split}.csv"
    df = pd.read_csv(RAW_DIR / fname, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Compute features
    features = compute_features(df)
    fingerprint = _compute_fingerprint(df)

    # Save parquet
    out_path = STORE_DIR / f"features_{split}_{FEATURE_VERSION}.parquet"
    features.to_parquet(out_path, index=False)
    print(f"  Saved features: {out_path} ({len(features):,} rows, {len(features.columns)} cols)")

    # Save registry JSON
    registry_path = STORE_DIR / "registry.json"
    manifest = {
        "version": FEATURE_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "split": split,
        "n_rows": len(features),
        "n_features": len(features.columns) - 2,  # exclude loan_id, month_index
        "input_fingerprint": fingerprint,
        "features": FEATURE_REGISTRY,
        "feature_columns": [c for c in features.columns if c not in ("loan_id", "month_index")],
        "files": {split: str(out_path.name)},
    }

    # Merge with existing registry if it exists
    if registry_path.exists():
        existing = json.loads(registry_path.read_text())
        existing.setdefault("files", {}).update(manifest["files"])
        existing["updated_at"] = manifest["created_at"]
        manifest = existing
        manifest["files"][split] = str(out_path.name)

    with open(registry_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Registry updated: {registry_path}")

    return out_path


def load_feature_store(split: str = "train") -> pd.DataFrame:
    """Load pre-computed features from the feature store."""
    path = STORE_DIR / f"features_{split}_{FEATURE_VERSION}.parquet"
    if not path.exists():
        print(f"[feature_store] Cache miss for {split} — building now...")
        build_feature_store(split)
    return pd.read_parquet(path)


if __name__ == "__main__":
    for split in ["train", "test"]:
        build_feature_store(split)

    # Verify
    train_features = load_feature_store("train")
    print(f"\n[feature_store] Loaded train features: {train_features.shape}")
    print(f"Feature columns: {[c for c in train_features.columns if c not in ('loan_id', 'month_index')]}")
