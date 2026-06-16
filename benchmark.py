import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# 🚀 Import the Big Three
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

DB_PATH = "data/research_engine.db"

def get_joined_features():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT s.date, s.ticker, s.close_price, s.volume, c.crude_oil, 
           c.copper_price, c.aluminium_price, m.bond_yield_10y, m.usd_index, m.cpi_inflation
    FROM stock_prices s
    INNER JOIN commodity_prices c ON s.date = c.date
    INNER JOIN macro_indicators m ON s.date = m.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    df = df.ffill()
    
    df['future_return_30d'] = df.groupby('ticker')['close_price'].shift(-30) / df['close_price'] - 1
    for lag in [30, 60]:
        df[f'crude_oil_lag_{lag}'] = df.groupby('ticker')['crude_oil'].shift(lag)
        df[f'bond_yield_lag_{lag}'] = df.groupby('ticker')['bond_yield_10y'].shift(lag)
        df[f'usd_index_lag_{lag}'] = df.groupby('ticker')['usd_index'].shift(lag)
        
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    print("📋 Fetching data and preparing benchmark matrices...")
    df = get_joined_features()
    
    feature_cols = [
        'close_price', 'volume', 'copper_price', 'aluminium_price', 'crude_oil',
        'bond_yield_10y', 'usd_index', 'cpi_inflation',
        'crude_oil_lag_30', 'bond_yield_lag_30', 'usd_index_lag_30',
        'crude_oil_lag_60', 'bond_yield_lag_60', 'usd_index_lag_60'
    ]
    
    X = df[feature_cols]
    y = df['future_return_30d']
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Initialize competitive models with default/comparable setups
    models = {
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost Regressor": CatBoostRegressor(iterations=100, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    results = {name: [] for name in models.keys()}
    
    print("🏋️ Running chronological validation loops across boosting frameworks...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            results[name].append(mae)
            
    print("\n⚡ ================= ACCURACY BENCHMARK RESULTS ================= ⚡")
    print(f"{'Model Framework':<25} | {'Mean Absolute Prediction Error (MAE %)':<30}")
    print("-" * 65)
    for name, scores in results.items():
        mean_mae_pct = np.mean(scores) * 100
        print(f"{name:<25} | {mean_mae_pct:.2f}%")
    print("=" * 65)