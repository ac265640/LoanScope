# Deploying the Drift Monitoring Dashboard to Streamlit Community Cloud

The drift dashboard (`src/monitoring/drift_dashboard.py`) is self-contained and
**requires zero manual data setup** — it auto-generates a representative sample
dataset on first load if `data/raw/` is empty (as it will be on a fresh clone,
since raw CSVs are gitignored).

---

## Prerequisites

- A GitHub account with this repository pushed to it.
- A [Streamlit Community Cloud](https://share.streamlit.io) account (free, sign in with GitHub).

---

## Step-by-step deployment

1. **Go to https://share.streamlit.io** and sign in with your GitHub account.

2. **Click "New app"** (top-right button).

3. **Connect your repository:**
   - Repository: `<your-github-username>/loan-performance-intelligence-engine`
   - Branch: `main`
   - Main file path: `src/monitoring/drift_dashboard.py`

4. **Set the requirements file** — Streamlit Cloud auto-detects `requirements.txt`
   at repo root, but this project provides a lighter file to keep build times short.
   Under **"Advanced settings"**:
   - Change the requirements path to: `requirements-dashboard.txt`
   
   > **Why a separate file?**  `requirements.txt` installs the full pipeline
   > (LightGBM, SHAP, MLflow, OpenAI, etc.) which is unnecessary for the dashboard
   > and would significantly slow the Streamlit Cloud build.  `requirements-dashboard.txt`
   > installs only the 6 packages actually needed (~60 s build vs. ~5–8 min).

5. **Secrets** — **None required.**  
   The drift dashboard does not use any LLM API keys, database connections, or
   external services.  Leave the Secrets section empty.

6. **Click "Deploy".**  First build takes 1–3 minutes.

7. **Wait for the app to load.**  On the very first request to a fresh deployment:
   - The app detects that `data/raw/` is empty (fresh clone — CSVs are gitignored).
   - It automatically runs the lite data generator:
     **7,000 loans × 24 months** (~168K rows) — enough for real drift statistics
     while fitting within free-tier RAM (~1 GB).
   - A spinner and progress message are shown during this ~30–60 second generation step.
   - Once generated, data is cached for the lifetime of the deployment instance —
     subsequent page loads are instant.

8. **Verify the app** by checking:
   - [ ] The demo-mode info banner is visible (confirms self-generation worked).
   - [ ] PSI bar chart renders with colored PASS/WARN/FAIL bars.
   - [ ] KS statistic chart renders for numeric features.
   - [ ] Full metrics table shows all 13 features without errors.

9. **Copy the live URL** (format: `https://<app-name>-<random>.streamlit.app`).
   This is the URL to submit in the "Demo Link" field of the submission form.

---

## Running locally (full scale)

```bash
# 1. Clone and set up
git clone <repo-url>
cd loan-performance-intelligence-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate full-scale data (50,000 loans × 36 months)
make run-all          # or: python src/data_generation/generate.py

# 3. Launch dashboard (full scale — no demo banner)
streamlit run src/monitoring/drift_dashboard.py
```

To test lite/demo mode locally:
```bash
DASHBOARD_LITE=1 streamlit run src/monitoring/drift_dashboard.py
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Build fails with `ModuleNotFoundError` | Wrong requirements file selected | In Advanced settings, confirm path is `requirements-dashboard.txt` |
| Spinner runs > 3 minutes with no result | Generator timeout on cold start | Re-deploy; if persistent, reduce `LITE_N_LOANS` in `drift_dashboard.py` to 4000 |
| Charts render but show all PASS (PSI ≈ 0) | Very small sample generated | Increase `LITE_N_LOANS` to 10000 for more meaningful drift signals |
| "Data files not found" error on plain `python` | Running as script, not streamlit | Use `streamlit run src/monitoring/drift_dashboard.py` |
