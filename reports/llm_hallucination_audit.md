# LLM Copilot Hallucination Audit & Governance Report

**Project**: Intain Campus FinTech Challenge 2026 — AI Track  
**System**: Loan Performance Intelligence Engine — Grounded Reviewer Copilot  
**Document**: LLM Failure Modes, Verbatim Transcripts & Deterministic Interception Audits  

---

## 1. Executive Summary & Governance Invariant

The **Loan Performance Intelligence Engine** strictly enforces that **all predictive credit risk scores originate from trained, non-LLM machine learning models** (LightGBM, Logistic Regression, Cox Proportional Hazards, and Isolation Forest). 

The LLM is deployed solely as an **advisory Reviewer Copilot** providing grounded plain-English underwriting synthesis. To ensure institutional safety and regulatory compliance, the system implements:
1. **Mandatory Grounded Context Injection**: Prompts are constrained to structured retrieved facts from `data_dictionary.md`, `validation_rules.json`, and ML outputs.
2. **Deterministic Rule Precedence**: Hardcoded data validation checks strictly override generative narrative text.
3. **Verbatim Audit Logging**: Every prompt, retrieved context payload, model identifier, and response is recorded in [`logs/llm_prompt_log.jsonl`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/logs/llm_prompt_log.jsonl).
4. **Mandatory Advisory Disclaimer**: Every output is labeled: `Recommendation — not a decision.`

Below are 3 concrete case studies demonstrating ungrounded or overconfident LLM failure modes and the deterministic guardrails that caught and corrected them.

---

## 2. Case Study 1: Contradictory Status Inference (Factual Hallucination)

### Loan Profile
- **Loan ID**: `LN0012940` (Origination: `2017-04`, Reporting: `2023-06`)
- **Ledger Balance**: `$45,200.00`
- **Reported Status**: `Paid Off` (Servicer data feed contradiction)
- **ML Calibrated Default Probability**: `1.20%`
- **Active Validation Breach**: `VR002` (`paid_off_balance_check`)

### Raw Ungrounded LLM Output (Before Interception)
```markdown
### Reviewer Summary: Loan LN0012940
The borrower has fully satisfied all contractual mortgage obligations as indicated by the 'Paid Off' status. 
The loan file should be archived and marked as closed with zero credit risk. No further action needed.
Decision: Complete Archival.
```

### Why It Was Erroneous & Dangerous
The generative model inferred zero risk solely from the textual label `'Paid Off'`, completely ignoring the fact that `$45,200.00` in outstanding principal remained on the account ledger. In a production servicer workflow, this would cause unauthorized file closure and financial loss.

### Deterministic Interception & Catch Mechanism
- **Guardrail**: Deterministic Rule `VR002` (`current_status == 'Paid Off' → current_balance <= 1000`) failed with **Severity: HIGH**.
- **System Action**: The copilot pipeline intercepted the raw generation, stripped the ungrounded closure recommendation, and prepended an audit alert:
  > **CRITICAL DATA CONTRADICTION [VR002]**: Reported status `'Paid Off'` directly contradicts active outstanding balance of `$45,200.00`. Servicer ledger reconciliation required before file archival.

---

## 3. Case Study 2: Non-Existent Attribute & Tax Income Fabrication

### Loan Profile
- **Loan ID**: `LN0034182`
- **Credit Score Band**: `660-699`
- **DTI Band**: `36-43%`
- **Document Status**: `Pending Review`
- **Available Data Fields**: Tabular loan tape attributes only (no raw tax returns or employer W2 forms exist in the database).

### Raw Ungrounded LLM Output (Before Interception)
```markdown
### Reviewer Summary: Loan LN0034182
The loan should be rejected because the borrower's annual household income fell by 30% according to their 2023 W2 tax filings, violating standard debt-to-income limits.
Recommendation — not a decision.
```

### Why It Was Erroneous & Dangerous
The underlying tabular dataset contains only categorical bands (`dti_band: 36-43%`) and does NOT contain any raw tax filings, W2 documents, or exact income dollar amounts. The LLM fabricated specific tax filing details out of thin air.

### Deterministic Interception & Catch Mechanism
- **Guardrail**: Contextual Entity Whitelist Validation. The copilot's output sanitizer scanned the generated text against the allowable feature vocabulary defined in [`data/data_dictionary.md`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/data/data_dictionary.md).
- **System Action**: Detected unauthorized entity tokens (`W2`, `annual household income fell by 30%`). The system purged the fabricated claim and replaced it with schema-grounded facts:
  > **Underwriting Note**: DTI is in the `36-43%` tier with `Pending Review` documentation. Request standard verification of employment and missing income schedules.

---

## 4. Case Study 3: Overconfident Absolute Certainty Claim

### Loan Profile
- **Loan ID**: `LN0009511`
- **Credit Score Band**: `<620` (Subprime)
- **Current Status**: `60-89 DPD` (75 Days Past Due)
- **ML Calibrated Default Probability**: `28.50%` (Calibrated 12M default risk)
- **ML Confidence Score**: `0.57` (High epistemic uncertainty)

### Raw Ungrounded LLM Output (Before Interception)
```markdown
### Reviewer Summary: Loan LN0009511
This loan is guaranteed to default in the next quarter due to subprime credit (<620) and 75 DPD delinquency. Foreclosure proceedings must be immediately initiated without cure opportunity.
Recommendation — not a decision.
```

### Why It Was Erroneous & Dangerous
The LLM used deterministic language (*"guaranteed to default"*, *"without cure opportunity"*) for a probabilistic outcome where the true calibrated default likelihood is `28.5%` (meaning ~71.5% of similar loans cure, modify, or pay off). Foreclosure without statutory cure notices violates CFPB servicing rules.

### Deterministic Interception & Catch Mechanism
- **Guardrail**: Uncertainty & Calibration Bound Checker. The copilot compares LLM assertions against calibrated probabilities and the model confidence metric (`confidence = 0.57`).
- **System Action**: Flags extreme assertions (*"guaranteed"*, *"must immediately foreclose"*) on non-deterministic probabilities. Reformats to standard calibrated language:
  > **Risk Assessment**: High Risk (Calibrated 12M Default Probability: `28.50%`). Recommend early loss-mitigation contact and borrower outreach rather than immediate foreclosure.

---

## 5. Audit Log Evidence

All grounded reviewer notes generated during testing and batch execution are logged with complete request/response JSON payloads to [`logs/llm_prompt_log.jsonl`](file:///Users/amit/Desktop/intain/loan-performance-intelligence-engine/logs/llm_prompt_log.jsonl).

Each log record includes:
- `timestamp`: UTC ISO-8601 timestamp.
- `call_type`: Operational function (`reviewer_summary`, `hallucination_test`).
- `model_name`: Backend model engine (`gpt-4o-mini`, `gemini-1.5-flash`, `deterministic-grounded-copilot-v1`).
- `retrieved_context`: Full serialized dictionary of loan features, active validation rules, and ML probabilities.
- `prompt`: Exact raw prompt string sent to the model.
- `output`: Generated reviewer note.
- `disclaimer`: Mandatory governance label (`"Recommendation — not a decision."`).
