"""
LightGBM training pipeline for the Macro Research Engine.

Loads real market data from research_engine.db, engineers features via the
shared features.py module, trains with walk-forward TimeSeriesSplit validation,
and saves production model + SHAP artifacts for the dashboard.

Outputs (saved to data/models/):
    - macro_lgb_model.pkl      : Serialized production LightGBM model
    - feature_cols.json         : Ordered feature column list (for inference alignment)
    - feature_importance.csv    : Feature importance rankings
    - metrics.json              : Cross-validation performance summary
"""

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

from features import load_joined_data, engineer_features, DB_PATH, ASSET_TICKERS

# Output paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "macro_lgb_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_pipeline():
    print("📊 Loading joined data from research_engine.db...")
    df = load_joined_data(DB_PATH)
    print(f"   Raw rows: {len(df):,} across {df['ticker'].nunique()} assets")

    print("🔧 Engineering features (momentum, volatility, lags, one-hot)...")
    df, feature_cols, macro_input_cols = engineer_features(df, target_horizons=(30, 60, 90))
    print(f"   Feature-ready rows: {len(df):,}")
    print(f"   Features: {len(feature_cols)}")

    # Primary target: 30-day forward return
    target_col = 'target_return_30d'
    X = df[feature_cols]
    y = df[target_col]

    print(f"   Training matrix: {X.shape}")

    # =============================================
    # Walk-Forward Cross-Validation (3-fold)
    # =============================================
    print("\n⏱️ Running walk-forward cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    baseline_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        model.fit(X_train, y_train)

        fold_r2 = model.score(X_test, y_test)
        cv_scores.append(fold_r2)

        # Zero-return baseline comparison
        baseline_ss_res = np.sum((y_test - 0) ** 2)
        baseline_ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
        baseline_r2 = 1 - (baseline_ss_res / baseline_ss_tot)
        baseline_scores.append(baseline_r2)

        print(f"   Fold {fold + 1}: R² = {fold_r2:.4f} | Baseline R² = {baseline_r2:.4f}")

    mean_cv_r2 = float(np.mean(cv_scores))
    mean_baseline_r2 = float(np.mean(baseline_scores))
    print(f"\n   Mean CV R²:       {mean_cv_r2:.4f}")
    print(f"   Mean Baseline R²: {mean_baseline_r2:.4f}")

    # =============================================
    # Multi-Horizon Comparison (informational)
    # =============================================
    print("\n📐 Multi-horizon R² comparison (last fold)...")
    horizon_r2 = {}
    for h in [30, 60, 90]:
        col = f'target_return_{h}d'
        if col in df.columns:
            y_h = df[col].iloc[test_idx]
            y_h = y_h.dropna()
            if len(y_h) > 0:
                X_h = X.iloc[y_h.index]
                r2 = model.score(X_h, y_h)
                horizon_r2[f'{h}d'] = round(r2, 4)
                print(f"   {h}-day horizon R²: {r2:.4f}")

    # =============================================
    # Train Production Model (full dataset)
    # =============================================
    print("\n🏗️ Training production model on full dataset...")
    prod_model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    prod_model.fit(X, y)

    # =============================================
    # SHAP Explainability
    # =============================================
    print("🔍 Computing SHAP expected value...")
    explainer = shap.TreeExplainer(prod_model)
    
    # explainer.expected_value can be a scalar, array, or list
    raw_ev = explainer.expected_value
    if hasattr(raw_ev, "__len__"):
        shap_expected_value = float(raw_ev[0])
    else:
        shap_expected_value = float(raw_ev)

    # =============================================
    # Save All Artifacts
    # =============================================
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model binary
    joblib.dump(prod_model, MODEL_PATH)
    print(f"   ✅ Model saved: {MODEL_PATH}")

    # Feature column order (critical for inference alignment)
    with open(FEATURE_COLS_PATH, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'macro_input_cols': macro_input_cols,
            'tickers': ASSET_TICKERS,
        }, f, indent=2)
    print(f"   ✅ Feature config saved: {FEATURE_COLS_PATH}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': prod_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    print(f"   ✅ Feature importance saved: {IMPORTANCE_PATH}")

    # Metrics summary
    metrics = {
        'mean_cv_r2': round(mean_cv_r2, 4),
        'mean_baseline_r2': round(mean_baseline_r2, 4),
        'shap_expected_value': round(shap_expected_value, 6),
        'n_features': len(feature_cols),
        'n_training_rows': len(X),
        'n_assets': df['ticker'].nunique(),
        'horizon_r2': horizon_r2,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved: {METRICS_PATH}")

    print("\n🔋 Training pipeline complete.")


if __name__ == "__main__":
    train_pipeline()