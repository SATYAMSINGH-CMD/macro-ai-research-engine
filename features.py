"""
Shared feature engineering module for the Macro Research Engine.

This is the single source of truth for all feature transformations.
Used by: trainmodel.py, benchmark.py, backtest.py, app.py

Guarantees that training features and inference features are identical.
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "data/research_engine.db"

# Canonical column names matching database.py schema
MACRO_COLS = [
    'bond_yield_10y', 'usd_index', 'cpi_inflation',
    'crude_oil', 'copper_price', 'aluminium_price'
]

# Full asset universe: 11 individual stocks + 6 sector ETFs
ASSET_TICKERS = sorted([
    # Semiconductors
    'NVDA', 'AMD', 'INTC',
    # Tech
    'AAPL', 'MSFT',
    # Automotive
    'TSLA', 'F', 'GM',
    # Airlines
    'DAL', 'UAL', 'AAL',
    # Sector & Index ETFs
    'QQQ', 'SPY', 'IWM', 'XLE', 'XLF', 'SMH'
])


def load_joined_data(db_path=DB_PATH):
    """Load and join stock_prices + commodity_prices + macro_indicators from SQLite.

    Returns:
        pd.DataFrame: Raw joined data sorted by (ticker, date).
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT s.date, s.ticker, s.close_price, s.volume,
           c.crude_oil, c.copper_price, c.aluminium_price,
           m.bond_yield_10y, m.usd_index, m.cpi_inflation
    FROM stock_prices s
    INNER JOIN commodity_prices c ON s.date = c.date
    INNER JOIN macro_indicators m ON s.date = m.date
    ORDER BY s.ticker, s.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Forward-fill any gaps within each ticker block
    df[MACRO_COLS] = df.groupby('ticker')[MACRO_COLS].ffill()
    df[MACRO_COLS] = df.groupby('ticker')[MACRO_COLS].bfill()

    return df


def engineer_features(df, target_horizons=(30, 60, 90)):
    """Transform raw joined data into model-ready features.

    Replaces raw close_price with momentum and volatility features.
    Adds macro/commodity lags and one-hot ticker encoding.

    Args:
        df: Raw joined DataFrame from load_joined_data().
        target_horizons: Tuple of forward-return horizons in trading days.

    Returns:
        df: DataFrame with features and targets appended.
        feature_cols: Ordered list of feature column names.
        macro_input_cols: List of macro columns controllable by dashboard sliders.
    """
    df = df.copy()

    # --- Price-derived features (replace raw close_price) ---
    df['daily_return'] = df.groupby('ticker')['close_price'].pct_change()
    df['return_5d'] = df.groupby('ticker')['close_price'].pct_change(5)
    df['return_20d'] = df.groupby('ticker')['close_price'].pct_change(20)
    df['volatility_20d'] = df.groupby('ticker')['daily_return'].transform(
        lambda x: x.rolling(20).std()
    )

    # --- Macro/commodity lag features ---
    for col in MACRO_COLS:
        df[f'{col}_lag_30'] = df.groupby('ticker')[col].shift(30)
        df[f'{col}_lag_60'] = df.groupby('ticker')[col].shift(60)

    # --- Target variables (multiple horizons) ---
    for h in target_horizons:
        df[f'target_return_{h}d'] = (
            df.groupby('ticker')['close_price'].shift(-h) / df['close_price'] - 1.0
        )

    # --- One-hot ticker encoding (sorted for deterministic column order) ---
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='asset', dtype=int)
    # Ensure all expected tickers have columns, even if absent from data
    for t in ASSET_TICKERS:
        col_name = f'asset_{t}'
        if col_name not in ticker_dummies.columns:
            ticker_dummies[col_name] = 0
    # Sort columns for deterministic order
    dummy_cols = sorted([f'asset_{t}' for t in ASSET_TICKERS])
    ticker_dummies = ticker_dummies[dummy_cols]
    df = pd.concat([df, ticker_dummies], axis=1)

    # --- Assemble feature column list (strict ordering) ---
    price_features = ['return_5d', 'return_20d', 'volatility_20d']
    macro_current = MACRO_COLS.copy()
    lag_features = []
    for col in MACRO_COLS:
        lag_features.append(f'{col}_lag_30')
        lag_features.append(f'{col}_lag_60')

    feature_cols = price_features + macro_current + lag_features + dummy_cols

    # Drop rows with NaN from rolling/lag/target calculations
    required_cols = feature_cols + [f'target_return_{target_horizons[0]}d']
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df, feature_cols, MACRO_COLS


def build_inference_row(db_path, ticker, feature_cols, macro_overrides=None):
    """Build a single feature vector for live prediction from latest available data.

    Args:
        db_path: Path to SQLite database.
        ticker: Stock ticker to predict for.
        feature_cols: Exact ordered list of feature columns (from training).
        macro_overrides: Dict of {column_name: value} to override current macro
                        values (e.g., from dashboard sliders).

    Returns:
        row_df: Single-row DataFrame with model-ready features in training order.
        metadata: Dict with latest_date, current_price, default_macro values.
    """
    conn = sqlite3.connect(db_path)

    # Pull enough rows for momentum (20d) + volatility (20d) + lags (60d)
    query = """
    SELECT s.date, s.ticker, s.close_price, s.volume,
           c.crude_oil, c.copper_price, c.aluminium_price,
           m.bond_yield_10y, m.usd_index, m.cpi_inflation
    FROM stock_prices s
    INNER JOIN commodity_prices c ON s.date = c.date
    INNER JOIN macro_indicators m ON s.date = m.date
    WHERE s.ticker = ?
    ORDER BY s.date DESC
    LIMIT 200
    """
    df = pd.read_sql_query(query, conn, params=[ticker])
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = df.sort_values('date').reset_index(drop=True)

    # Capture metadata before any overrides
    latest_date = df['date'].iloc[-1]
    current_price = float(df['close_price'].iloc[-1])
    default_macro = {col: float(df[col].iloc[-1]) for col in MACRO_COLS}

    # Apply macro overrides to the latest row only
    if macro_overrides:
        for col, val in macro_overrides.items():
            if col in MACRO_COLS:
                df.loc[df.index[-1], col] = val

    # --- Compute price-derived features ---
    df['daily_return'] = df['close_price'].pct_change()
    df['return_5d'] = df['close_price'].pct_change(5)
    df['return_20d'] = df['close_price'].pct_change(20)
    df['volatility_20d'] = df['daily_return'].rolling(20).std()

    # --- Compute macro lag features ---
    for col in MACRO_COLS:
        df[f'{col}_lag_30'] = df[col].shift(30)
        df[f'{col}_lag_60'] = df[col].shift(60)

    # --- One-hot ticker encoding ---
    for t in ASSET_TICKERS:
        df[f'asset_{t}'] = 1 if t == ticker else 0

    # Extract the latest row and select features in the exact training order
    latest = df.iloc[[-1]].copy()

    metadata = {
        'latest_date': latest_date,
        'current_price': current_price,
        'default_macro': default_macro,
    }

    return latest[feature_cols], metadata
