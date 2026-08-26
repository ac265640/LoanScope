# Human-in-the-Loop Active Learning Report

## ⚠️ Simulation Disclosure

Reviewer feedback in this report is **simulated** for the hackathon demonstration.
A simulated reviewer model (with 85% accuracy) provides accept/reject/correct decisions.
In production deployment, a real reviewer UI (e.g., Streamlit annotation interface)
would replace the simulated reviewer, collecting decisions from qualified domain experts.

---

## 1. Active Learning Loop

```

Round 1: Train Isolation Forest → Flag top 10% anomalous loans

         ↓

Reviewer: Accept / Reject / Correct each flagged case

         ↓

Round 2: Adjust decision threshold based on reviewer's FP feedback → Re-evaluate

```

## 2. Reviewer Feedback Summary

| Decision | Count |
|----------|-------|
| ✅ Accept | 303 |
| ❌ Reject (False Positive) | 197 |
| 🔄 Correct (Type Change) | 51 |
| **Total Reviewed** | **500** |

**False Positive Rate in initial batch:** 39.4%

## 3. Before vs After Performance

| Metric | Before Feedback | After Feedback | Change |
|--------|----------------|---------------|--------|
| AUC | 1.0 | 1.0 | +0.0000 |
| Precision | 0.336 | 0.4179 | +0.0819 |
| Recall | 1.0 | 1.0 | +0.0000 |
| F1 | 0.503 | 0.5895 | +0.0865 |

Flagged count reduced from **500** to **402**
by raising threshold from default to 0.5949.

## 4. Production Integration Design

In a real deployment, the active learning loop would:

1. Present flagged loans in a **Streamlit annotation UI** with loan details and SHAP explanation.
2. Collect structured feedback: accept/reject + corrected exception type.
3. Store feedback in a PostgreSQL table with reviewer ID and timestamp.
4. Periodically retrain the Isolation Forest / exception classifier on confirmed cases.
5. Track reviewer agreement rate as a quality metric for the annotation process.

---

_Script: `src/models/anomaly/active_learning.py` | Advanced Feature #14_
