# Project: Fraud Forensics Pipeline

## Overview
Portfolio project targeting the **Deloitte Analytic & Forensic Technology** junior position (Madrid). End-to-end fraud detection pipeline with an interactive Streamlit dashboard designed as an investigator's tool. Must demonstrate domain-aware analytical thinking alongside tool-building capability.

## Core Principle
Every analytical decision must be justified through the lens of: **"What would matter to someone investigating fraud?"** This is a forensic analytics project, not a generic ML exercise.

## Dataset
- **Source:** kartik2112/fraud-detection from Kaggle (Sparkov-generated synthetic data)
- **Cited in:** Amazon's Fraud Dataset Benchmark (FDB) research
- **Files:** `data/raw/fraudTrain.csv` and `data/raw/fraudTest.csv`
- **Size:** ~1.3M transactions (pre-split train + test)
- **Target variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)
- **Key columns:** trans_date_trans_time, amt, merchant, category, gender, lat, long, merch_lat, merch_long, city_pop, job, dob, is_fraud
- **Columns to drop before modeling:** first, last, street, cc_num, trans_num, unix_time (identifiers, not features)
- **CRITICAL:** This dataset is entirely synthetic. All narrative must frame findings as methodological demonstrations, never as real-world fraud insights.

## Technical Stack
- **Python 3.11**
- **Notebook EDA/plots:** matplotlib, seaborn (static, clean, readable)
- **Dashboard visuals:** plotly (interactive, investigator-friendly)
- **ML:** scikit-learn (LogisticRegression — interpretable), XGBoost (performant)
- **Imbalance:** imbalanced-learn (SMOTE on training data only)
- **Dashboard:** Streamlit
- **Persistence:** joblib for models, CSV/JSON for dashboard data

## Repository Structure
```
fraud-detection-forensic-analytics/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── processed/  
│   └── raw/                 # Original CSVs (gitignored)
├── dashboard_data/           # Pre-computed outputs for Streamlit (committed)
│   ├── predictions.csv       # Test set with risk scores and predictions
│   ├── model_metrics.json    # Pre-computed evaluation metrics
│   └── pr_curve_data.csv     # Precision-recall curve data points
├── notebooks/
│   └── analysis.ipynb        # EDA, modeling, evaluation
├── app/
│   └── dashboard.py          # Streamlit app (3 views only)
├── src/
│   ├── data_processing.py    # Data loading and feature engineering
│   ├── model.py              # Model training and prediction
│   └── evaluation.py         # Metrics and visualizations
├──scripts/
│   └── downlaod_data.py      # Downloads dataset from kaggle using API, README.md must include explanation with alternative method (URL)			
├── models/                   # Saved .joblib files (gitignored)
└── screenshots/              # Dashboard screenshots for README
```

## Analytical Decisions (Already Made)
- **Train/test split:** Use the pre-split files from Kaggle as-is
- **Imbalance handling:** SMOTE applied to training data only; justify in notebook
- **Model 1 (interpretable):** Logistic Regression — show coefficients, explain which features drive risk
- **Model 2 (performant):** XGBoost — show feature importances, compare with Model 1
- **Evaluation priority:** Recall over precision — missing fraud (FN) is far costlier than false alarms (FP)
- **Threshold:** Sweep multiple thresholds, recommend one with justification framed as investigator capacity
- **Feature engineering targets:** customer-merchant distance (from lat/long), hour of day, day of week, log-amount, customer age from dob

## Dashboard Architecture
- **Exactly 3 views.** No fourth view regardless of ideas during development.
- Dashboard loads **pre-computed data only** — never retrains models or reads raw dataset
- Data source: `dashboard_data/` folder with predictions CSV and metrics JSON
- **View 1 — Overview:** KPIs, fraud rate, distributions by category/time/amount
- **View 2 — Transaction Explorer:** Filterable table with risk scores, the core investigator tool
- **View 3 — Model Performance:** Confusion matrix, PR curves, model comparison table
- Deploy to **Streamlit Community Cloud** for a live demo URL

## Coding Standards
- Use `joblib` for model persistence
- Use `plotly` for dashboard visuals; `matplotlib`/`seaborn` for notebook EDA
- Include a **"Forensic Note"** in comments explaining *why* a feature or metric matters for an investigator
- Print intermediate results frequently: shapes, value counts, sanity checks
- Keep notebook cells short and focused — one concept per cell
- Every code cell should be accompanied by a markdown cell explaining the "why"

## Workflow Instructions
- When modifying existing code, provide **only the changed snippets** with explanation — no full-file rewrites unless explicitly asked
- When creating new files, provide the complete file
- If something is ambiguous, make the choice that best serves the forensic investigation framing
- If a decision hasn't been specified above, ask before assuming