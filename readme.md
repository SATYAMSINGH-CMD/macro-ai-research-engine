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
├── pipeline.py                 <-- Automated ETL data ingestion script (integrates FRED CSV)
├── features.py                 <-- Shared feature engineering module (training & inference alignment)
├── trainmodel.py               <-- Model training pipeline (LightGBM + SHAP expected values)
├── benchmark.py                <-- Multi-model accuracy tournament (LightGBM vs XGBoost vs CatBoost)
├── backtest.py                 <-- Leakage-free walk-forward expanding window strategy simulator
└── app.py                      <-- Streamlit explorer with real model, database, and SHAP waterfall

---

## 🏃 Setup & Execution Order

To set up the database, download the market data, train the production model, and run the Streamlit dashboard:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database Schema:**
   ```bash
   python database.py
   ```

3. **Ingest Market & Macro Data:**
   Fetches stock, commodity, and Treasury yield data via `yfinance`, and daily inflation expectations (`T10YIE`) directly from FRED's CSV endpoint (no API key required).
   ```bash
   python pipeline.py
   ```

4. **Train Model & Compute SHAP Expected Values:**
   Joins data, engineers features via `features.py`, trains LightGBM under walk-forward validation, and saves production model + metadata artifacts.
   ```bash
   python trainmodel.py
   ```

5. **Run Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

6. **(Optional) Run Benchmark & Backtest Validation:**
   ```bash
   python benchmark.py
   python backtest.py
   ```