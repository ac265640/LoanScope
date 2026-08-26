# Survival Analysis Report — Competing-Risk Model

## 1. Event Summary

| Outcome | Count | % of Portfolio |
|---------|-------|----------------|
| Default | 80 | 14.4% |
| Prepaid | 47 | 8.5% |
| Censored | 428 | 77.1% |
| **Total** | **555** | 100.0% |

## 2. Global Cumulative Incidence Functions (Aalen-Johansen)

The CIF represents the probability of experiencing a specific event before time `t`,
**accounting for** the competing risk (the other event type).

| Horizon | CIF(Default) | CIF(Prepaid) | Sum |
|---------|-------------|--------------|-----|
| 12 months | 0.0847 | 0.0631 | 0.1477 |
| 36 months | 0.1441 | 0.0847 | 0.2288 |

**Note:** CIF(Default) + CIF(Prepaid) + P(Censored) = 1.0, which is the fundamental
constraint of the competing-risk framework.

## 3. CIF by Credit Band

| Credit Band | N Loans | 12m CIF Default | 12m CIF Prepaid | 36m CIF Default | 36m CIF Prepaid |
|------------|---------|----------------|----------------|----------------|----------------|
| 620-659 | 71 | 0.1690 | 0.0141 | 0.2817 | 0.0141 |
| 660-699 | 90 | 0.1000 | 0.0111 | 0.1889 | 0.0222 |
| 700-739 | 127 | 0.0394 | 0.0787 | 0.0472 | 0.0866 |
| 740-779 | 98 | 0.0306 | 0.0714 | 0.1020 | 0.1122 |
| 780+ | 80 | 0.0250 | 0.1750 | 0.0500 | 0.2250 |
| <620 | 46 | 0.1957 | 0.0000 | 0.3261 | 0.0000 |
| nan | 43 | 0.1628 | 0.0465 | 0.1860 | 0.0930 |

## 4. Competing-Risk vs Single-Risk Comparison

The single-risk Kaplan-Meier model treats competing events as random censoring,
leading to **upward bias** in the estimated probability of each event type.
The Aalen-Johansen competing-risk CIF corrects this.

|   horizon_months |   competing_risk_CIF_default |   single_risk_CIF_default |   bias_default |   competing_risk_CIF_prepaid |   single_risk_CIF_prepaid |   bias_prepaid |
|-----------------:|-----------------------------:|--------------------------:|---------------:|-----------------------------:|--------------------------:|---------------:|
|               12 |                       0.0847 |                    0.0889 |         0.0043 |                       0.0631 |                    0.0683 |         0.0053 |
|               24 |                       0.1315 |                    0.1738 |         0.0423 |                       0.0793 |                    0.1023 |         0.023  |
|               36 |                       0.1441 |                    0.2313 |         0.0872 |                       0.0847 |                    0.1412 |         0.0565 |

**Key insight:** The single-risk model systematically over-estimates the probability
of each event by treating the competing event as mere censoring. For example,
the default CIF bias (`single_risk - competing_risk`) is positive at all horizons,
meaning a naive KM analysis would over-state the default risk in portfolios with
active prepayments (common in low-rate environments).

## 5. Methodology

- **Estimator:** Non-parametric Aalen-Johansen estimator for CIF curves.
- **Events:** Default (event_type=1) includes delinquency states (30/60/90 DPD,
  charge-off); Prepaid (event_type=2) includes full payoff and voluntary prepayment.
- **Censoring:** Loans still active at dataset end treated as right-censored.
- **Segmentation:** Credit band based on `credit_score` at origination.
- **Comparison model:** Cause-specific Kaplan-Meier (single-risk) treating
  competing event as non-informative censoring (yields upward-biased CIF).

## 6. Implementation

- Script: `src/models/survival/competing_risk.py`
- Run: `python -m src.models.survival.competing_risk`
- Advanced Feature #1 of 15
