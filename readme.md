# 📊 Macroeconomic AI Research Engine

A full-stack quantitative data pipeline and predictive analytics application that uses tree-based machine learning frameworks to simulate, explain, and backtest 30-day future stock returns based on shifting global macroeconomic indicators and industrial commodity prices.

---

## 🚀 Key Architectural Features
* **Production ETL Data Pipeline:** Ingests daily financial data using `yfinance` with automated fallback bypass routing to handle API constraints, realigns volatile time-series steps, and drops market holiday gaps using `pandas`.
* **Relational SQL Storage Layer:** Designed a robust SQLite database schema equipped with strict composite primary keys (`date` + `ticker`) to guarantee relational data integrity.
* **Explainable AI (XAI) Integration:** Leverages single-prediction **SHAP (SHapley Additive exPlanations)** values to isolate live marginal indicator contributions, mapping exactly why the AI makes specific bullish or bearish forecasts.
* **Historical Backtest Sandbox:** Features an interactive trading simulator using a custom date-range filter and an asset allocation threshold rule, applying a strict 1-day execution delay to eliminate look-ahead bias.
* **Multi-Model Tournament Benchmarking:** Evaluates performance models by pitting **LightGBM vs. XGBoost vs. CatBoost** side-by-side using chronological `TimeSeriesSplit` cross-validation to eliminate data leakage.

---

## 🛠️ Tech Stack & Frameworks
* **Language:** Python 3.13
* **Database Layer:** SQLite3
* **Machine Learning:** LightGBM, XGBoost, CatBoost, Scikit-Learn
* **Explainability:** SHAP (TreeExplainer)
* **Frontend Web UI:** Streamlit
* **Visualization:** Matplotlib, Pandas Plotting

---

## 📂 Repository Blueprint
```text
macro research engine/
│
├── data/
│   ├── research_engine.db      <-- Local SQLite DB (Populated via pipeline.py)
│   └── models/
│       └── macro_lgb_model.pkl <-- Serialized binary model parameters
│
├── database.py                 <-- DB schema initializer & constraint builder
├── pipeline.py                 <-- Automated ETL data ingestion script
├── trainmodel.py               <-- SQL multi-table join and initial model training
├── benchmark.py                <-- Validation script pitting the Big Three boosting models
├── backtest.py                 <-- Historical strategy compounding ledger loop
└── app.py                      <-- Two-way Streamlit dashboard user interface