import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from sklearn.model_selection import TimeSeriesSplit

DB_PATH = "data/research_engine.db"
MODEL_DIR = "data/models"

def load_and_merge_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        s.date,
        s.ticker,
        s.close_price,
        s.volume,
        c.copper_price,
        c.aluminium_price,
        c.crude_oil,
        m.bond_yield_10y,
        m.usd_index,
        m.cpi_inflation
    FROM stock_prices s
    INNER JOIN commodity_prices c ON s.date = c.date
    INNER JOIN macro_indicators m ON s.date = m.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"📈 Raw matrix joined successfully! Pulled {len(df)} total dataset rows.")
    return df

def engineer_features(df):
    print("🧼 Processing data matrix and engineering features...")
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    df = df.ffill()
    
    df['future_return_30d'] = df.groupby('ticker')['close_price'].shift(-30) / df['close_price'] - 1
    
    for lag in [30, 60]:
        df[f'crude_oil_lag_{lag}'] = df.groupby('ticker')['crude_oil'].shift(lag)
        df[f'bond_yield_lag_{lag}'] = df.groupby('ticker')['bond_yield_10y'].shift(lag)
        df[f'usd_index_lag_{lag}'] = df.groupby('ticker')['usd_index'].shift(lag)
        
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Feature engineering complete. Matrix shape: {df.shape}")
    return df

def train_predictive_model(df):
    print("🤖 Initializing machine learning pipeline...")
    feature_cols = [
        'close_price', 'volume', 'copper_price', 'aluminium_price', 'crude_oil',
        'bond_yield_10y', 'usd_index', 'cpi_inflation',
        'crude_oil_lag_30', 'bond_yield_lag_30', 'usd_index_lag_30',
        'crude_oil_lag_60', 'bond_yield_lag_60', 'usd_index_lag_60'
    ]
    target_col = 'future_return_30d'
    
    X = df[feature_cols]
    y = df[target_col]
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    print("🏋️ Training LightGBM Regressor across rolling historical windows...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"   📊 Fold {fold + 1} R² Prediction Score: {score:.4f}")
        
    print(f"✅ Training loop complete. Mean R² Score: {np.mean(scores):.4f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = os.path.join(MODEL_DIR, "macro_lgb_model.pkl")
    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
        
    print(f"💾 Trained model brain successfully serialized and saved to {model_file}")
    return model

if __name__ == "__main__":
    print("🚀 Starting Model Training Script execution...")
    raw_df = load_and_merge_data()
    processed_df = engineer_features(raw_df)
    final_model = train_predictive_model(processed_df)
    print("\n🏁 Process finished! Your AI model is trained and ready for the Streamlit dashboard.")