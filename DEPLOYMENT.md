# Deploying the LoanScope Multi-Page Showcase App to Streamlit Community Cloud

The LoanScope showcase application (`src/monitoring/app.py`) is a comprehensive multi-page
system walkthrough covering credit risk prediction, calibration, survival analysis,
anomaly detection, macroeconomic stress scenarios, and feature drift surveillance.

It is self-contained and **requires zero manual data setup** — the drift monitoring module
auto-generates a representative sample dataset in <1s on first load if `data/raw/` is empty
(as it will be on a fresh clone, since raw CSVs are gitignored).

---

## Architecture of the Hosted Showcase

The app follows Streamlit's native `pages/` convention:

```
src/monitoring/
├── app.py                     # Main landing page & navigation hub
└── pages/
    ├── 1_Overview.py          # System architecture, data lineage, headline KPIs
    ├── 2_Predictions.py       # Multi-horizon GBDT models, calibration diagrams, threshold trade-offs
    ├── 3_Survival_and_Risk.py # Competing risks Aalen-Johansen CIF vs Kaplan-Meier (+8.72pp bias)
    ├── 4_Anomaly_Cases.py     # 25 reviewer cases, Rule Engine vs Learned ML, audit notes
    ├── 5_Scenario_Simulator.py # Macro stress scenarios & Monte Carlo fan charts
    └── 6_Drift_Monitoring.py  # Feature distribution stability (PSI & KS metrics)
```

---

## Step-by-Step Deployment (New App or Existing App Update)

### Option A: Updating an Existing Streamlit Cloud App
If you already deployed the app:
1. Go to your app at **`loanscope-drift.streamlit.app`** (or your app URL).
2. Click **"Manage app"** (bottom-right) → **"Settings"** (gear icon) → **"General"**.
3. Change **Main file path** to: `src/monitoring/app.py`.
4. Click **"Save"** (the app will reload automatically with sidebar navigation to all 6 pages).

---

### Option B: Deploying a Fresh App
1. **Go to https://share.streamlit.io** and sign in with your GitHub account.
2. **Click "New app"** (top-right button).
3. **Connect your repository:**
   - Repository: `<your-github-username>/loan-performance-intelligence-engine` (or `LoanScope`)
   - Branch: `main`
   - **Main file path:** `src/monitoring/app.py`
4. **Set the requirements file**:
   - Click **"Advanced settings"**
   - Set requirements path to: `requirements-dashboard.txt` (or leave default `requirements.txt`)
5. **Secrets**: **None required** (leave empty).
6. **Click "Deploy"**.
7. Once live, test each page via the left sidebar.
8. Copy the live URL (format: `https://<app-name>-<random>.streamlit.app`) for your hackathon submission.

---

## Running Locally (Full Scale)

```bash
# 1. Clone and set up virtual environment
git clone <repo-url>
cd loan-performance-intelligence-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run full data generation & pipeline (50,000 loans x 36 months)
make run-all

# 3. Launch full multi-page showcase app
streamlit run src/monitoring/app.py
```

---

## Security & Performance Governance

- **Zero Live LLM Calls**: The public deployment does not invoke live LLM endpoints or require API keys. Reviewer copilot notes are static pre-logged samples from `logs/llm_prompt_log.jsonl`.
- **Zero Heavy Live Computation**: All model metrics, calibration bins, scenario projections, and Monte Carlo curves read from pre-computed artifacts.
- **Free-Tier Friendly**: Memory usage stays well below the 1GB RAM ceiling.
