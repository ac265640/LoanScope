# Feature Drift Monitoring Report

## Portfolio Stability Summary

| Metric | Value |
|---|---|
| Overall Status | **FAIL** |
| Features Monitored | 32 |
| Stable Features | 22 |
| Moderate Drift | 1 |
| Severe Drift | 9 |

> PSI Thresholds: Stable < 0.10 | Moderate: 0.10–0.25 | Severe: > 0.25

## ⚠️ Severe Drift Features (PSI > 0.25) — Model Revalidation Recommended

- **state_freq**: PSI=5.3545, KS=0.4329, Mean shift=0.8% (Train μ=0.0500 → Test μ=0.0504)
- **servicer_name_freq**: PSI=12.3348, KS=0.4161, Mean shift=0.1% (Train μ=0.2000 → Test μ=0.2003)
- **loan_purpose_freq**: PSI=2.1556, KS=0.4019, Mean shift=2.2% (Train μ=0.2992 → Test μ=0.3058)
- **occupancy_type_freq**: PSI=1.9591, KS=0.7496, Mean shift=0.4% (Train μ=0.5918 → Test μ=0.5944)
- **property_type_freq**: PSI=10.2260, KS=0.6019, Mean shift=1.3% (Train μ=0.4067 → Test μ=0.4015)
- **document_status_freq**: PSI=15.8861, KS=0.5100, Mean shift=0.0% (Train μ=0.2500 → Test μ=0.2501)
- **source_system_freq**: PSI=15.8861, KS=0.5100, Mean shift=0.0% (Train μ=0.2500 → Test μ=0.2501)
- **orig_year**: PSI=12.0758, KS=1.0000, Mean shift=0.6% (Train μ=2011.0622 → Test μ=2022.3414)
- **is_legacy_vintage**: PSI=7.2698, KS=0.5228, Mean shift=100.0% (Train μ=0.5228 → Test μ=0.0000)

## ⚡ Moderate Drift Features (PSI 0.10–0.25) — Monitor Closely

- **credit_score_ordinal**: PSI=0.1709, KS=0.0718, Mean shift=6.9%

## Top 15 Features by PSI

| Feature | PSI | KS Stat | Status | Mean Shift % |
|---|---|---|---|---|
| document_status_freq | 15.8861 | 0.5100 | Severe Drift | 0.0% |
| source_system_freq | 15.8861 | 0.5100 | Severe Drift | 0.0% |
| servicer_name_freq | 12.3348 | 0.4161 | Severe Drift | 0.1% |
| orig_year | 12.0758 | 1.0000 | Severe Drift | 0.6% |
| property_type_freq | 10.2260 | 0.6019 | Severe Drift | 1.3% |
| is_legacy_vintage | 7.2698 | 0.5228 | Severe Drift | 100.0% |
| state_freq | 5.3545 | 0.4329 | Severe Drift | 0.8% |
| loan_purpose_freq | 2.1556 | 0.4019 | Severe Drift | 2.2% |
| occupancy_type_freq | 1.9591 | 0.7496 | Severe Drift | 0.4% |
| credit_score_ordinal | 0.1709 | 0.0718 | Moderate Drift | 6.9% |
| ever_delinquent_past | 0.0051 | 0.0211 | Stable | 19.6% |
| interest_rate_imputed | 0.0045 | 0.0588 | Stable | 1.0% |
| rate_to_market_spread | 0.0045 | 0.0588 | Stable | 7.0% |
| original_balance | 0.0038 | 0.0143 | Stable | 0.0% |
| current_balance | 0.0028 | 0.0124 | Stable | 0.1% |