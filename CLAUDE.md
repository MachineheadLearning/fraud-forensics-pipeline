# Project: Fraud Forensics Pipeline

## Overview
Portfolio project End-to-end fraud detection pipeline with an interactive Streamlit dashboard designed as an investigator's tool. 

## Dataset
- **Source:** kartik2112/fraud-detection from Kaggle (Sparkov-generated synthetic data)
- **Cited in:** Amazon's Fraud Dataset Benchmark (FDB) research
- **Files:** `data/raw/fraudTrain.csv` and `data/raw/fraudTest.csv`
- **Size:** ~1.3M transactions (pre-split train + test)
- **Target variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)
- **Key columns:** trans_date_trans_time, amt, merchant, category, gender, lat, long, merch_lat, merch_long, city_pop, job, dob, is_fraud
- **Columns to drop before modeling:** first, last, street, cc_num, trans_num, unix_time (identifiers, not features)
- **CRITICAL:** This dataset is entirely synthetic. All narrative must frame findings as methodological demonstrations, not as real-world fraud insights.

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
├──scripts/
│   └── downlaod_data.py      # Downloads dataset from kaggle using API, README.md must include explanation with alternative method (URL)			
├── models/                   # Saved .joblib files (gitignored)
└── screenshots/              # Dashboard screenshots for README
```
