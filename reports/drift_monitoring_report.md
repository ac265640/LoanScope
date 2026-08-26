# Feature Drift Monitoring Report

## Train vs Test Distribution Drift (PSI / KS)

| Feature | Type | PSI | KS | Status |
|---------|------|-----|-----|--------|
| loan_age_months | numeric | 0.0009 | 0.0136 | PASS |
| remaining_term_months | numeric | 0.0011 | 0.0096 | PASS |
| original_balance | numeric | 0.0034 | 0.0152 | PASS |
| current_balance | numeric | 0.0028 | 0.0135 | PASS |
| interest_rate | numeric | 0.0046 | 0.0291 | PASS |
| days_past_due | numeric | 0.0000 | 0.0088 | PASS |
| credit_score_band | categorical | 0.1445 | nan | WARN |
| ltv_band | categorical | 0.0028 | nan | PASS |
| dti_band | categorical | 0.0010 | nan | PASS |
| state | categorical | 0.0094 | nan | PASS |
| loan_purpose | categorical | 0.0026 | nan | PASS |
| occupancy_type | categorical | 0.0000 | nan | PASS |
| current_status | categorical | 0.0049 | nan | PASS |

**Thresholds:** PSI < 0.10 = PASS, 0.10–0.25 = WARN, > 0.25 = FAIL

**Launch dashboard:** `streamlit run src/monitoring/drift_dashboard.py`