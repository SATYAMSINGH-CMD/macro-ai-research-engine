"""
Multi-model benchmark: LightGBM vs XGBoost vs CatBoost.

Uses shared features.py for consistent feature engineering.
Evaluates all three boosting frameworks on the same walk-forward
TimeSeriesSplit to produce a fair MAE comparison.
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from features import load_joined_data, engineer_features, DB_PATH


if __name__ == "__main__":
    print("📋 Loading data and engineering features...")
    df = load_joined_data(DB_PATH)
    df, feature_cols, _ = engineer_features(df)

    X = df[feature_cols]
    y = df['target_return_30d']

    print(f"   Matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    tscv = TimeSeriesSplit(n_splits=3)

    models = {
        "XGBoost":  xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.05, random_state=42, verbose=0),
    }

    results = {name: [] for name in models}

    print("🏋️ Running chronological validation across boosting frameworks...\n")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            results[name].append(mae)

        print(f"   Fold {fold + 1} complete")

    print("\n⚡ ================= ACCURACY BENCHMARK RESULTS ================= ⚡")
    print(f"{'Model':<15} | {'Mean MAE (%)':>15}")
    print("-" * 35)
    for name, scores in results.items():
        mean_mae_pct = np.mean(scores) * 100
        print(f"{name:<15} | {mean_mae_pct:>14.2f}%")
    print("=" * 35)