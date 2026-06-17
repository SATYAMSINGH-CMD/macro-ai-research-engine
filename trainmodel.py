import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# Structural path declarations
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "macro_research.db"
MODEL_PATH = BASE_DIR / "models" / "macro_lgb_model.pkl"
IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

def load_and_preprocess_data():
    print("Loading data from database...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT p.date, p.ticker, p.close_price, m.usd_index, m.crude_oil, m.yield_10y, m.copper, m.cpi, m.aluminum
        FROM asset_prices p
        LEFT JOIN macro_indicators m ON p.date = m.date
        ORDER BY p.date ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Isolate forward fill boundaries within unique ticker blocks
    macro_cols = ['usd_index', 'crude_oil', 'yield_10y', 'copper', 'cpi', 'aluminum']
    df[macro_cols] = df.groupby('ticker')[macro_cols].ffill()
    df[macro_cols] = df.groupby('ticker')[macro_cols].bfill()
    
    print("Engineering target variables and lag features...")
    # Target return calculation window
    df['target_30d_return'] = df.groupby('ticker')['close_price'].shift(-30) / df['close_price'] - 1.0
    
    # Extract historical macro factor lags
    for col in macro_cols:
        df[f'{col}_lag_30'] = df.groupby('ticker')[col].shift(30)
        df[f'{col}_lag_60'] = df.groupby('ticker')[col].shift(60)
        
    # FIX: Implement One-Hot Encoding to eliminate ordinal sector bias
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='asset', dtype=int)
    df = pd.concat([df, ticker_dummies], axis=1)
    
    # Drop rows containing rolling padding artifacts
    df = df.dropna().reset_index(drop=True)
    return df, ticker_dummies.columns.tolist()

def train_pipeline():
    df, dummy_cols = load_and_preprocess_data()
    
    # Isolate explicit input arrays
    lag_cols = [col for col in df.columns if '_lag_' in col]
    feature_cols = dummy_cols + lag_cols
    target_col = 'target_30d_return'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Total training matrix shape: {X.shape}")
    
    # Time-Series Walk-Forward Validation Setup
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    baseline_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        fold_score = model.score(X_test, y_test)
        cv_scores.append(fold_score)
        
        # Zero-Return Baseline out-of-sample benchmark calculations
        y_baseline_pred = np.zeros_like(y_test)
        baseline_ss_res = np.sum((y_test - y_baseline_pred) ** 2)
        baseline_ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
        baseline_r2 = 1 - (baseline_ss_res / baseline_ss_tot)
        baseline_scores.append(baseline_r2)
        
        print(f"Fold {fold+1} - Out-of-Sample R²: {fold_score:.4f} | Zero-Return Baseline R²: {baseline_r2:.4f}")
    
    mean_cv_r2 = float(np.mean(cv_scores))
    mean_baseline_r2 = float(np.mean(baseline_scores))
    
    print(f"Mean Validation R²: {mean_cv_r2:.4f}")
    print(f"Mean Baseline R²: {mean_baseline_r2:.4f}")
    
    # Train production model on complete chronological dataset slice
    print("Training production model...")
    prod_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    prod_model.fit(X, y)
    
    # Save the feature importance profile matrix
    print("Saving feature importance map...")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": prod_model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    # Ensure persistence output storage structures exist safely
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    joblib.dump(prod_model, MODEL_PATH)
    
    # FIX: Export the true cross-validation evaluation summary into metrics.json
    metrics_summary = {
        "mean_cv_r2": round(mean_cv_r2, 4),
        "mean_baseline_r2": round(mean_baseline_r2, 4)
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("Pipeline compilation complete.")

if __name__ == "__main__":
    train_pipeline()