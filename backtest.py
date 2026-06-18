"""
Walk-forward backtest simulation for NVDA.

Fixes the critical leakage issue: instead of train-on-all → predict-on-all,
this uses an expanding-window approach where the model only ever predicts
on data it has never seen during training.

Flow per window:
    1. Train on all data BEFORE the test window
    2. Predict on the test window only (unseen)
    3. Generate trading signals from out-of-sample predictions
    4. Track compounded returns with a 1-day execution delay
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features import load_joined_data, engineer_features, DB_PATH


def run_backtest(ticker='NVDA', n_splits=4):
    """Run a proper walk-forward backtest with expanding training windows."""

    print(f"⏳ Running walk-forward backtest for {ticker}...")

    # Load and engineer features using the shared module
    df = load_joined_data(DB_PATH)
    df, feature_cols, _ = engineer_features(df)

    # Filter to the target ticker
    df = df[df['ticker'] == ticker].copy().reset_index(drop=True)

    if len(df) < 200:
        print(f"❌ Not enough data for {ticker} ({len(df)} rows). Need at least 200.")
        return

    df['daily_return'] = df['close_price'].pct_change().fillna(0)

    X = df[feature_cols]
    y = df['target_return_30d']

    models = {
        "XGBoost":  xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.05, random_state=42, verbose=0),
    }

    # ================================================
    # Expanding-Window Walk-Forward Split
    # ================================================
    total_rows = len(df)
    # Reserve first 60% for initial training, split remainder into n_splits test windows
    initial_train_end = int(total_rows * 0.6)
    test_rows = total_rows - initial_train_end
    window_size = test_rows // n_splits

    strategy_signals = {name: np.zeros(total_rows) for name in models}

    print(f"   Total rows: {total_rows:,}")
    print(f"   Initial training window: 0 → {initial_train_end}")
    print(f"   Test windows: {n_splits} × {window_size} rows\n")

    for window in range(n_splits):
        test_start = initial_train_end + (window * window_size)
        test_end = min(test_start + window_size, total_rows)

        if test_start >= total_rows:
            break

        # Expanding window: train on everything BEFORE the test period
        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Signal: go long if predicted 30d return > 2%
            signal = np.where(preds > 0.02, 1, 0)
            strategy_signals[name][test_start:test_end] = signal

        print(f"   Window {window + 1}: train[0:{test_start}] → test[{test_start}:{test_end}]")

    # ================================================
    # Compute Strategy Returns
    # ================================================
    # Buy-and-hold benchmark
    bh_growth = (1 + df['daily_return']).cumprod()
    final_bh = (bh_growth.iloc[-1] - 1) * 100

    strategy_results = {}
    for name, signals in strategy_signals.items():
        # Apply 1-day execution delay to eliminate look-ahead bias
        allocation = pd.Series(signals).shift(1).fillna(0).values
        strat_daily = df['daily_return'] * allocation
        final_growth = (1 + strat_daily).cumprod().iloc[-1]
        strategy_results[name] = (final_growth - 1) * 100

    # ================================================
    # Print Results
    # ================================================
    print(f"\n📊 ================= WALK-FORWARD BACKTEST: {ticker} ================= 📊")
    print(f"🗓️  Period: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"📏  Test region: last {test_rows} trading days ({n_splits} expanding windows)")
    print("-" * 65)
    print(f"{'Buy & Hold':<25} | {final_bh:+.2f}%")
    for name, ret in strategy_results.items():
        alpha = ret - final_bh
        print(f"{name + ' Strategy':<25} | {ret:+.2f}%  (α = {alpha:+.2f}%)")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()