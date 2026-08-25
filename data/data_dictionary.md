# Loan Performance Intelligence Engine — Data Dictionary

This data dictionary serves as the single source of truth for tabular schema definitions, business rules, valid value domains, and data governance semantics across the Loan Performance Intelligence Engine. It is also used directly as retrieved context to ground the LLM Reviewer Copilot.

---

## 1. Identifiers & Temporal Fields

| Field Name | Type | Description | Valid Range / Allowed Values | Business Rules & Invariants |
| :--- | :--- | :--- | :--- | :--- |
| `loan_id` | String | Unique alphanumeric identifier for each loan account. | Format: `LN[0-9]{7}` (e.g. `LN0000001`) | Primary key for loan level static entities. Never changes over the loan lifecycle. |
| `month_index` | Integer | Sequence number of the observation month relative to loan origination. | `1` to `360` (typical max `36` in panel) | `1` represents the first monthly reporting period immediately post-origination. Strictly strictly monotonically increasing. |
| `reporting_month` | String (YYYY-MM) | Calendar year and month when loan performance data was reported by the servicer. | `2003-01` to `2026-06` | Invariant: `reporting_month >= origination_month`. Any row where `reporting_month < origination_month` is an anomaly. |
| `origination_month` | String (YYYY-MM) | Calendar year and month when the loan was initially funded/originated. | `2003-01` to `2023-06` | Static across all monthly reporting cycles for a given `loan_id`. |
| `loan_age_months` | Integer | Elapsed age of the loan in months since origination. | `1` to `360` | Invariant: `loan_age_months == month_index` under standard monthly reporting cadence. |
| `remaining_term_months` | Integer | Number of scheduled amortization months remaining until contractual loan maturity. | `0` to `360` | Decreases by 1 each standard cycle. Reaches `0` at contractual maturity. |
| `last_updated_at` | Date (YYYY-MM-DD) | Timestamp of the latest servicer or ingestion system update record. | Valid ISO-8601 date string | Must be on or after the first day of the corresponding `reporting_month`. |

---

## 2. Financial & Credit Attributes

| Field Name | Type | Description | Valid Range / Allowed Values | Business Rules & Invariants |
| :--- | :--- | :--- | :--- | :--- |
| `original_balance` | Float | Total principal balance disbursed at loan closing ($ USD). | `$50,000` to `$2,000,000` | Static origination attribute. Constant across all panel observations. |
| `current_balance` | Float | Outstanding unpaid principal balance (UPB) as of reporting date ($ USD). | `$0.00` to `$2,000,000+` | Invariant: For `current_status == 'Paid Off'` or `'Prepaid'`, `current_balance` must be `$0` (or `<= $1,000` threshold). Cannot exceed `2.0 * original_balance` without formal negative amortization exception. |
| `interest_rate` | Float | Annualized note interest rate expressed as a percentage (e.g., `4.5` = 4.5% p.a.). | `2.0` to `12.0` (Outliers: `15.0+`) | Note interest rate. Used for scheduled monthly payment amortization calculations. |
| `credit_score_band` | Categorical | FICO credit score tier evaluated at origination. | `<620`, `620-659`, `660-699`, `700-739`, `740-779`, `780+` | Primary underwriting creditworthiness metric. Prone to Missing-Not-At-Random (MNAR) patterns in legacy pre-2010 vintages. |
| `ltv_band` | Categorical | Loan-to-Value ratio band at origination (Original Balance / Appraised Value). | `<60%`, `60-70%`, `70-80%`, `80-90%`, `90-95%`, `>95%` | High LTV (>80%) typically indicates higher default loss severity and elevated default propensity unless mitigated by PMI. |
| `dti_band` | Categorical | Debt-to-Income ratio band of borrower at origination (Total Monthly Debt / Gross Monthly Income). | `<20%`, `20-28%`, `28-36%`, `36-43%`, `>43%` | Cash-flow debt service capacity metric. Ratios `>43%` indicate non-conforming / QM threshold stress. |

---

## 3. Property & Underwriting Characteristics

| Field Name | Type | Description | Valid Range / Allowed Values | Business Rules & Invariants |
| :--- | :--- | :--- | :--- | :--- |
| `state` | Categorical | Two-letter US state postal code of the collateral property. | 20 core US States (CA, TX, FL, NY, IL, PA, OH, GA, NC, MI, NJ, VA, WA, AZ, MA, TN, IN, MO, MD, WI) | Geographical risk proxy. Subject to state-level foreclosure moratoriums, climate risks, and macro housing price index (HPI) trends. |
| `loan_purpose` | Categorical | Reason for loan financing at origination. | `Purchase`, `Refinance`, `Cash-Out Refinance`, `Home Improvement` | `Cash-Out Refinance` historically carries higher credit risk than rate/term `Refinance` and `Purchase`. |
| `occupancy_type` | Categorical | Intended occupancy profile of the mortgaged residential property. | `Primary`, `Second Home`, `Investment` | `Investment` and `Second Home` carry higher default risk and lower servicer cure priority than `Primary` residence. |
| `property_type` | Categorical | Structural architectural type of the underlying residential real estate collateral. | `Single Family`, `Condo`, `Multi-Family`, `Townhouse`, `Manufactured` | `Manufactured` and `Condo` often have distinct depreciation and resale liquidity dynamics compared to `Single Family`. |
| `servicer_name` | Categorical | Designated primary master loan servicing entity. | `Servicer_A`, `Servicer_B`, `Servicer_C`, `Servicer_D`, `Servicer_E` | Operational performance, modification protocols, and data reconciliation feeds vary by servicer entity. |
| `source_system` | Categorical | Origination or core servicing technological platform. | `CoreLogic`, `Black Knight`, `Ellie Mae`, `Encompass` | Tracks data lineage and ingestion feed origin. |
| `document_status` | Categorical | Status of loan documentation trailing file and audit verification. | `Complete`, `Pending Review`, `Missing Items`, `Under Exception` | Gaps (`Missing Items`) combined with loan modifications trigger compliance/audit exceptions. |

---

## 4. Current State & Delinquency Tracking

| Field Name | Type | Description | Valid Range / Allowed Values | Business Rules & Invariants |
| :--- | :--- | :--- | :--- | :--- |
| `current_status` | Categorical | Performance state of the loan at current reporting month. | `Current`, `30-59 DPD`, `60-89 DPD`, `90+ DPD`, `Default`, `Prepaid`, `Paid Off` | Terminal states: `Default`, `Prepaid`, `Paid Off`. Transitional states: `30-59 DPD`, `60-89 DPD`, `90+ DPD`. |
| `days_past_due` | Integer | Cumulative number of calendar days elapsed since contractual payment due date. | `0` to `500+` | `0` for Current/Prepaid/Paid Off. For `Default`, `days_past_due` must generally be `>= 60` or `>= 90`. |
| `modification_flag` | Binary Integer | Flag indicating whether the loan terms have undergone formal loan modification or workout. | `0` (No), `1` (Yes) | Invariant: Once set to `1`, remains `1` or reflects active restructured loan profile. Requires audit verification. |
| `prepayment_flag` | Binary Integer | Flag indicating whether the loan principal was curtailed or paid in full ahead of schedule. | `0` (No), `1` (Yes) | Terminal state trigger: transitions status to `Prepaid` and sets `current_balance = 0`. |
| `default_flag` | Binary Integer | Flag indicating formal credit default / charge-off / liquidation event. | `0` (No), `1` (Yes) | Invariant: `current_status == 'Default' → default_flag == 1`. |
| `loss_severity_band` | Categorical | Estimated or realized net loss percentage upon liquidation/foreclosure. | `None`, `<20%`, `20-35%`, `35-50%`, `50-65%`, `>65%` | Populated primarily when `default_flag == 1` or upon credit liquidation. |

---

## 5. Supervised Target Variables (Prediction Outcomes)

| Target Name | Type | Horizon | Description & Modeling Nuance |
| :--- | :--- | :--- | :--- |
| `next_3m_delinquency_flag` | Binary (0/1) | Forward 3 Months | Indicates whether the loan enters delinquency (`30-59 DPD`, `60-89 DPD`, `90+ DPD`, or `Default`) at any point in the next 3 observation months. Used for short-term early warning surveillance. |
| `next_6m_delinquency_flag` | Binary (0/1) | Forward 6 Months | Indicates whether delinquency occurs across the upcoming 6-month window. Medium-term credit watchlist trigger. |
| `next_12m_default_flag` | Binary (0/1) | Forward 12 Months | Indicates whether formal loan default occurs within the subsequent 12 months. Primary credit risk loss forecasting target. Highly imbalanced (~2% - 6%). |
| `next_12m_prepayment_flag` | Binary (0/1) | Forward 12 Months | Indicates whether voluntary prepayment in full occurs within the next 12 months. Critical for cash-flow duration and CPR (Conditional Prepayment Rate) modeling. |
| `next_state` | Multiclass Categorical | Forward 1 Month | Transition target representing the immediate next-month performance state (`Current`, `30-59 DPD`, `60-89 DPD`, `90+ DPD`, `Default`, `Prepaid`, `Paid Off`). |

---

## 6. Anomaly & Exception Variables

| Field Name | Type | Description | Target / Label Meaning |
| :--- | :--- | :--- | :--- |
| `exception_required` | Binary (0/1) | Review flag requiring operational intervention or secondary underwriter sign-off. | `1` indicates validation rule breach, extreme data conflict, or severe anomaly requiring human review. |
| `exception_type` | Categorical | Categorization of data quality / underwriting failure. | Allowed classes: `None`, `Missing Document`, `Valuation Discrepancy`, `Income Verification Gap`, `Data Conflict`, `Stale Record`, `Balance Inconsistency`, `Date Anomaly`. |
| `anomaly_score` | Float | Unsupervised outlier / inconsistency score computed by isolation forest or distance metrics. | Normalized `[0.0, 1.0]` where higher values indicate severe tabular or behavioural anomaly. |

---

## 7. LLM Grounding & Reviewer Governance Guidelines

1. **Recommendation Status**: All text, summaries, and action plans generated by LLMs are strictly categorized as **"Recommendation — not a decision."** Final credit or operational actions must be confirmed by authorized reviewers.
2. **Context Retrieval Injection**: When querying the LLM for loan assessment, the prompt MUST include:
   - Specific borrower static parameters (`credit_score_band`, `ltv_band`, `dti_band`, `state`).
   - Recent 6-month payment and balance trajectory.
   - Deterministic validation rule outputs and detected data conflicts.
   - Calibrated ML probabilities (`prob_default_12m`, `prob_delinq_3m`, `prob_prepay_12m`) and top SHAP feature drivers.
3. **Hallucination Safeguards**: The copilot system cross-checks all LLM textual assertions against deterministic `validation_rules.json` and tabular inputs before rendering outputs to the user.
