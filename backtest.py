import sqlite3
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

DB_PATH = "data/research_engine.db"

def load_backtest_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT s.date, s.ticker, s.close_price, s.volume, c.crude_oil, 
           c.copper_price, c.aluminium_price, m.bond_yield_10y, m.usd_index, m.cpi_inflation
    FROM stock_prices s
    INNER JOIN commodity_prices c ON s.date = c.date
    INNER JOIN macro_indicators m ON s.date = m.date
    WHERE s.ticker = 'NVDA'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = df.sort_values(by='date').reset_index(drop=True)
    df = df.ffill()
    
    for lag in [30, 60]:
        df[f'crude_oil_lag_{lag}'] = df['crude_oil'].shift(lag)
        df[f'bond_yield_lag_{lag}'] = df['bond_yield_10y'].shift(lag)
        df[f'usd_index_lag_{lag}'] = df['usd_index'].shift(lag)
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    print("⏳ Running Multi-Model Algorithmic Backtest Simulation for NVDA...")
    df = load_backtest_data()
    
    feature_cols = [
        'close_price', 'volume', 'copper_price', 'aluminium_price', 'crude_oil',
        'bond_yield_10y', 'usd_index', 'cpi_inflation',
        'crude_oil_lag_30', 'bond_yield_lag_30', 'usd_index_lag_30',
        'crude_oil_lag_60', 'bond_yield_lag_60', 'usd_index_lag_60'
    ]
    
    df['daily_asset_return'] = df['close_price'].pct_change()
    
    # Fit the variants inside the strategy loop
    X_train_full = df[feature_cols]
    y_train_lbl = df['daily_asset_return'].shift(-30).fillna(0) # Target baseline alignment
    
    models = {
        "XGBoost Strategy": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        "LightGBM Strategy": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost Strategy": CatBoostRegressor(iterations=100, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    # Compound benchmark returns
    df['buy_and_hold_growth'] = (1 + df['daily_asset_return'].fillna(0)).cumprod()
    
    print("🧱 Evaluating individual strategy return paths...")
    strategy_results = {}
    
    for name, model in models.items():
        # Train on historical vector space
        model.fit(X_train_full, y_train_lbl)
        preds = model.predict(X_train_full)
        
        # Generation allocation rules (+2% threshold trigger)
        signal = np.where(preds > 0.02, 1, 0)
        allocation = pd.Series(signal).shift(1).fillna(0).values
        
        strat_daily_return = df['daily_asset_return'].fillna(0) * allocation
        final_growth = (1 + strat_daily_return).cumprod().iloc[-1]
        strategy_results[name] = (final_growth - 1) * 100

    final_bh = (df['buy_and_hold_growth'].iloc[-1] - 1) * 100
    
    print("\n📊 ================= HISTORICAL BACKTEST RETURNS ================= 📊")
    print(f"🗓️ Simulation Window: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print("-" * 65)
    print(f"{'Buy & Hold Baseline':<25} | {final_bh:+.2f}%")
    for name, return_pct in strategy_results.items():
        print(f"{name:<25} | {return_pct:+.2f}%")
    print("=" * 65)