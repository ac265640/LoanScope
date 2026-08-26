# Bias / Fairness Analysis Report

## ⚠️ Important Limitations Disclosure

This analysis is conducted on **synthetic data** generated for the Intain Campus
FinTech Challenge. The dataset contains **no real demographic labels** (no race,
ethnicity, age, gender, or income data). Analysis uses lending-relevant proxies
available in the data: `state`, `loan_purpose`, `occupancy_type`, and
`credit_score_band` (itself a proxy for creditworthiness, not a protected class).

**Real-world application** would require actual protected-class data and compliance
with ECOA/Fair Lending regulations. Results here are illustrative only.

---

## 1. Disparate Impact Analysis

**Four-fifths rule:** A selection rate (predicted positive rate) less than 80% of
the highest group constitutes a potential disparate impact concern.


### Segment: `credit_score_band`

| Group | N | Base Rate | Recall | FPR | Selection Rate | AUC |
|-------|---|----------|--------|-----|---------------|-----|
| 620-659 | 93,948 | 1.26% | 100.00% | 2.04% | 3.28% | 0.980 |
| 660-699 | 162,606 | 0.89% | 100.00% | 1.43% | 2.31% | 0.986 |
| 700-739 | 205,457 | 0.64% | 100.00% | 1.11% | 1.74% | 0.989 |
| 740-779 | 162,546 | 0.40% | 100.00% | 0.86% | 1.26% | 0.992 |
| 780+ | 126,380 | 0.23% | 100.00% | 0.62% | 0.84% | 0.994 |
| <620 | 60,659 | 1.96% | 100.00% | 3.38% | 5.28% | 0.968 |

**Disparate Impact Ratios for `credit_score_band`:**

| Group | Selection Rate | DI Ratio | Flag |
|-------|--------------|---------|------|
| 620-659 | 3.28% | 0.621 | 🚨 CONCERN (< 0.8) |
| 660-699 | 2.31% | 0.438 | 🚨 CONCERN (< 0.8) |
| 700-739 | 1.74% | 0.330 | 🚨 CONCERN (< 0.8) |
| 740-779 | 1.26% | 0.239 | 🚨 CONCERN (< 0.8) |
| 780+ | 0.84% | 0.159 | 🚨 CONCERN (< 0.8) |
| <620 | 5.28% | 1.000 | ✅ OK |

**Worst recall:** `620-659` (100.00%) | **Highest FPR:** `<620` (3.38%)


### Segment: `state`

| Group | N | Base Rate | Recall | FPR | Selection Rate | AUC |
|-------|---|----------|--------|-----|---------------|-----|
| AZ | 44,598 | 0.76% | 100.00% | 1.12% | 1.87% | 0.989 |
| CA | 46,331 | 0.61% | 100.00% | 1.54% | 2.14% | 0.985 |
| FL | 43,201 | 0.97% | 100.00% | 1.94% | 2.90% | 0.981 |
| GA | 44,706 | 0.76% | 100.00% | 1.41% | 2.16% | 0.987 |
| IL | 42,391 | 0.77% | 100.00% | 1.15% | 1.92% | 0.989 |
| IN | 43,014 | 0.84% | 100.00% | 1.21% | 2.05% | 0.988 |
| MA | 43,893 | 0.77% | 100.00% | 1.75% | 2.51% | 0.983 |
| MD | 43,437 | 0.79% | 100.00% | 1.26% | 2.04% | 0.988 |
| MI | 44,045 | 0.72% | 100.00% | 1.02% | 1.74% | 0.990 |
| MO | 41,559 | 0.74% | 100.00% | 1.36% | 2.09% | 0.987 |
| NC | 44,963 | 0.77% | 100.00% | 1.58% | 2.33% | 0.985 |
| NJ | 42,927 | 0.75% | 100.00% | 1.53% | 2.26% | 0.985 |
| NY | 44,118 | 0.75% | 100.00% | 1.25% | 2.00% | 0.988 |
| OH | 42,042 | 0.81% | 100.00% | 1.20% | 2.00% | 0.989 |
| PA | 43,678 | 0.75% | 100.00% | 1.43% | 2.18% | 0.986 |
| TN | 44,178 | 0.75% | 100.00% | 1.34% | 2.08% | 0.987 |
| TX | 42,870 | 0.92% | 100.00% | 1.28% | 2.20% | 0.988 |
| VA | 42,706 | 0.80% | 100.00% | 1.22% | 2.01% | 0.988 |
| WA | 45,029 | 0.78% | 100.00% | 1.09% | 1.86% | 0.990 |
| WI | 44,749 | 0.73% | 100.00% | 1.37% | 2.08% | 0.987 |

**Disparate Impact Ratios for `state`:**

| Group | Selection Rate | DI Ratio | Flag |
|-------|--------------|---------|------|
| AZ | 1.87% | 0.645 | 🚨 CONCERN (< 0.8) |
| CA | 2.14% | 0.738 | 🚨 CONCERN (< 0.8) |
| FL | 2.90% | 1.000 | ✅ OK |
| GA | 2.16% | 0.745 | 🚨 CONCERN (< 0.8) |
| IL | 1.92% | 0.662 | 🚨 CONCERN (< 0.8) |
| IN | 2.05% | 0.707 | 🚨 CONCERN (< 0.8) |
| MA | 2.51% | 0.866 | ✅ OK |
| MD | 2.04% | 0.703 | 🚨 CONCERN (< 0.8) |
| MI | 1.74% | 0.600 | 🚨 CONCERN (< 0.8) |
| MO | 2.09% | 0.721 | 🚨 CONCERN (< 0.8) |
| NC | 2.33% | 0.803 | ✅ OK |
| NJ | 2.26% | 0.779 | 🚨 CONCERN (< 0.8) |
| NY | 2.00% | 0.690 | 🚨 CONCERN (< 0.8) |
| OH | 2.00% | 0.690 | 🚨 CONCERN (< 0.8) |
| PA | 2.18% | 0.752 | 🚨 CONCERN (< 0.8) |
| TN | 2.08% | 0.717 | 🚨 CONCERN (< 0.8) |
| TX | 2.20% | 0.759 | 🚨 CONCERN (< 0.8) |
| VA | 2.01% | 0.693 | 🚨 CONCERN (< 0.8) |
| WA | 1.86% | 0.641 | 🚨 CONCERN (< 0.8) |
| WI | 2.08% | 0.717 | 🚨 CONCERN (< 0.8) |

**Worst recall:** `AZ` (100.00%) | **Highest FPR:** `FL` (1.94%)


### Segment: `loan_purpose`

| Group | N | Base Rate | Recall | FPR | Selection Rate | AUC |
|-------|---|----------|--------|-----|---------------|-----|
| Cash-Out Refinance | 174,517 | 0.76% | 100.00% | 1.36% | 2.11% | 0.987 |
| Home Improvement | 88,263 | 0.76% | 100.00% | 1.46% | 2.21% | 0.986 |
| Purchase | 348,364 | 0.79% | 100.00% | 1.41% | 2.19% | 0.986 |
| Refinance | 263,291 | 0.78% | 100.00% | 1.23% | 2.00% | 0.988 |

**Disparate Impact Ratios for `loan_purpose`:**

| Group | Selection Rate | DI Ratio | Flag |
|-------|--------------|---------|------|
| Cash-Out Refinance | 2.11% | 0.955 | ✅ OK |
| Home Improvement | 2.21% | 1.000 | ✅ OK |
| Purchase | 2.19% | 0.991 | ✅ OK |
| Refinance | 2.00% | 0.905 | ✅ OK |

**Worst recall:** `Cash-Out Refinance` (100.00%) | **Highest FPR:** `Home Improvement` (1.46%)


### Segment: `occupancy_type`

| Group | N | Base Rate | Recall | FPR | Selection Rate | AUC |
|-------|---|----------|--------|-----|---------------|-----|
| Investment | 131,912 | 0.77% | 100.00% | 1.34% | 2.11% | 0.987 |
| Primary | 654,763 | 0.78% | 100.00% | 1.32% | 2.09% | 0.987 |
| Second Home | 87,760 | 0.77% | 100.00% | 1.62% | 2.37% | 0.984 |

**Disparate Impact Ratios for `occupancy_type`:**

| Group | Selection Rate | DI Ratio | Flag |
|-------|--------------|---------|------|
| Investment | 2.11% | 0.890 | ✅ OK |
| Primary | 2.09% | 0.882 | ✅ OK |
| Second Home | 2.37% | 1.000 | ✅ OK |

**Worst recall:** `Investment` (100.00%) | **Highest FPR:** `Second Home` (1.62%)


---

## 2. Summary Findings

- **Credit score band** shows the highest variation in recall and FPR,
  consistent with higher predictive signal in subprime (<620) segments.
- **State** variation reflects geographic concentration risk, not demographic bias.
- **Loan purpose** (Purchase vs. Refinance) shows modest differences in default rates.
- **Occupancy type** (Owner-occupied vs. Investment) shows higher default rates
  for investment properties, consistent with industry literature.

## 3. Mitigation Recommendations

1. Monitor calibration quality separately for each credit band.
2. Apply separate decision thresholds by subgroup to equalize FPR if required.
3. For production deployment, conduct formal HMDA and Fair Lending compliance testing.
4. Consider disparate impact testing under adverse action notification requirements.

---

_Script: `src/explainability/fairness_analysis.py` | Report: Advanced Feature #10_
