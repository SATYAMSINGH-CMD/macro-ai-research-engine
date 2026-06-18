import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from features import build_inference_row, ASSET_TICKERS

# Page configuration optimized for a clean, professional dashboard view
st.set_page_config(page_title="Macro Factor Impact Explorer", layout="wide")

# ==========================================
# STYLING SHEET: CLEAN INDUSTRIAL LIGHT THEME
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
    
    /* Remove standard Streamlit structural distractions */
    span[data-testid="stSidebarCollapseButton"], 
    div[data-testid="collapsedControl"],
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #2d2d2d;
        font-family: 'Inter', sans-serif;
    }
    
    /* Focus typography strictly on industry-standard slates */
    label, [data-testid="stWidgetLabel"] p {
        color: #1a2238 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    h1, h2, h3, h4, .mono-text {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Top Header Banner Component */
    .hero-header {
        background-color: #1a2238;
        color: #ffffff;
        padding: 3rem 2rem;
        text-align: center;
        border-bottom: 4px solid #c8a84b;
        margin: -6rem -4rem 2rem -4rem;
    }
    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #a0aabf;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Metric Display Cards */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #1a2238;
    }
    
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #e9ecef;
        font-size: 0.85rem;
        color: #2d2d2d;
    }
    .text-safe { color: #4a7c59; font-weight: 600; }
    .text-crit { color: #8f3f3f; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# Visual Hero Banner Layout
st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>MACRO FACTOR IMPACT EXPLORER</div>
        <div class='hero-subtitle'>Sector-specific factor sensitivity testing environment mapping asset return variances under custom macroeconomic stress scenarios.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD PRODUCTION MODEL AND ARTIFACTS
# ==========================================
from pathlib import Path

# Paths resolved relative to this app.py file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "models" / "macro_lgb_model.pkl"
CONFIG_PATH = BASE_DIR / "data" / "models" / "feature_cols.json"
METRICS_PATH = BASE_DIR / "data" / "models" / "metrics.json"
APP_DB_PATH = BASE_DIR / "data" / "research_engine.db"

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        with open(CONFIG_PATH, "r") as f:
            feature_config = json.load(f)
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return model, feature_config, metrics
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}. Make sure you run `trainmodel.py` first.")
        st.stop()

model, feature_config, metrics = load_model_artifacts()
feature_cols = feature_config["feature_cols"]
tickers = feature_config["tickers"]

# Asset target selector configuration
selected_ticker = st.selectbox("Asset Target", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
st.markdown("---")

# ==========================================
# FETCH BASELINE DATA AND INITIALIZE SCENARIO
# ==========================================
# Retrieve latest actual row from database to populate default slider values
@st.cache_data
def get_initial_row(ticker):
    try:
        row, meta = build_inference_row(APP_DB_PATH, ticker, feature_cols)
        return meta
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        st.stop()

meta = get_initial_row(selected_ticker)
default_macro = meta['default_macro']
current_price = meta['current_price']
latest_date = meta['latest_date']

# Helper to safely clip values to slider bounds
def get_slider_val(key, min_val, max_val, fallback):
    val = default_macro.get(key, fallback)
    return float(np.clip(val, min_val, max_val))

# ==========================================
# MAIN INTERACTIVE CORE LAYOUT GRID
# ==========================================
col_inputs, col_model, col_scenario = st.columns([30, 40, 30], gap="large")

# --- PANEL 1: INPUTS (LEFT) ---
with col_inputs:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Interactive Stress Scenario</h4>", unsafe_allow_html=True)
    st.caption(f"Pre-filled with latest market data ({latest_date})")
    
    sim_yield = st.slider("10Y Treasury Yield (%)", 0.50, 8.00, get_slider_val('bond_yield_10y', 0.50, 8.00, 4.42), 0.01)
    sim_usd = st.slider("US Dollar Index (DXY)", 70.00, 130.00, get_slider_val('usd_index', 70.00, 130.00, 100.00), 0.10)
    sim_cpi = st.slider("CPI Inflation Proxy (%)", -1.00, 10.00, get_slider_val('cpi_inflation', -1.00, 10.00, 3.20), 0.10)
    sim_oil = st.slider("Crude Oil Futures ($/bbl)", 20.00, 160.00, get_slider_val('crude_oil', 20.00, 160.00, 80.00), 0.50)
    sim_copper = st.slider("Copper Price ($/lb)", 1.00, 8.00, get_slider_val('copper_price', 1.00, 8.00, 4.00), 0.05)
    sim_aluminum = st.slider("Aluminum Price ($/ton)", 1000.0, 5000.0, get_slider_val('aluminium_price', 1000.0, 5000.0, 2500.0), 10.0)

# Build custom scenario input feature row
macro_overrides = {
    'bond_yield_10y': sim_yield,
    'usd_index': sim_usd,
    'cpi_inflation': sim_cpi,
    'crude_oil': sim_oil,
    'copper_price': sim_copper,
    'aluminium_price': sim_aluminum
}
live_input_df, _ = build_inference_row(APP_DB_PATH, selected_ticker, feature_cols, macro_overrides=macro_overrides)

# Compute live model inference prediction
pred_30d_return = model.predict(live_input_df)[0]
target_price = current_price * (1.0 + pred_30d_return)

# --- PANEL 2: MODEL ANALYTICS & RESULTS (CENTER) ---
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)
shap_values = explainer(live_input_df)

# Handle potential shape variation in SHAP versions
if hasattr(shap_values, "values"):
    shap_vals_array = shap_values.values[0]
else:
    shap_vals_array = shap_values[0]

# Custom matplotlib SHAP plotting helper
def plot_shap_contributions(shap_vals, feature_names, max_display=10):
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'val': shap_vals * 100.0  # Scale fractional returns to percentages
    })
    
    name_mapping = {
        'bond_yield_10y': '10Y Treasury Yield',
        'usd_index': 'US Dollar Index',
        'cpi_inflation': 'CPI Inflation Proxy',
        'crude_oil': 'Crude Oil Price',
        'copper_price': 'Copper Price',
        'aluminium_price': 'Aluminum Price',
        'return_5d': '5-Day Price Return',
        'return_20d': '20-Day Price Return',
        'volatility_20d': '20-Day Return Volatility'
    }
    for col in ['bond_yield_10y', 'usd_index', 'cpi_inflation', 'crude_oil', 'copper_price', 'aluminium_price']:
        name_mapping[f'{col}_lag_30'] = f'{name_mapping[col]} (30d Lag)'
        name_mapping[f'{col}_lag_60'] = f'{name_mapping[col]} (60d Lag)'
        
    df_shap['display_name'] = df_shap['feature'].apply(lambda x: name_mapping.get(x, x.replace('asset_', 'Asset: ')))
    
    # Filter out exact zeros and sort
    df_shap = df_shap[df_shap['val'].abs() > 0.001].copy()
    df_shap['abs_val'] = df_shap['val'].abs()
    df_shap = df_shap.sort_values('abs_val', ascending=True).tail(max_display)
    
    if df_shap.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No significant factor drivers found", ha='center', va='center')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')
    
    colors = ['#c8a84b' if x >= 0 else '#8f3f3f' for x in df_shap['val']]
    bars = ax.barh(df_shap['display_name'], df_shap['val'], color=colors, height=0.6, edgecolor='none')
    
    # Label value on bars
    for bar in bars:
        width = bar.get_width()
        label_x = width + 0.1 if width >= 0 else width - 0.1
        ha = 'left' if width >= 0 else 'right'
        ax.text(
            label_x, bar.get_y() + bar.get_height()/2, 
            f"{width:+.2f}%", 
            va='center', ha=ha, 
            color='#1a2238', 
            fontweight='bold',
            fontsize=8.5
        )
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#e9ecef')
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e9ecef')
    ax.set_axisbelow(True)
    
    plt.title("Key Return Drivers (SHAP Value Contribution)", fontsize=10.5, fontweight='bold', color='#1a2238', pad=15)
    plt.tight_layout()
    return fig

with col_model:
    st.markdown(f"<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Model Analytics: {selected_ticker}</h4>", unsafe_allow_html=True)
    
    # Real cross-validation metrics
    st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #c8a84b;'>
            <div class='metric-label'>Cross-Validation R² Score (Walk-Forward)</div>
            <div class='metric-value'>R² = {metrics['mean_cv_r2']:.4f}</div>
            <div style='font-size:0.8rem; color:#6c757d; margin-top:0.4rem;'>Benchmark (Zero-Return Model) R²: <b>{metrics['mean_baseline_r2']:.4f}</b></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Current Price [{selected_ticker}]</div>
            <div class='metric-value'>${current_price:.2f}</div>
            <div style='font-size:0.8rem; color:#6c757d; margin-top:0.4rem;'>Estimated 30-Day Target: <b>${target_price:.2f}</b></div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Model 30-Day Expected Return</div>
            <div class='metric-value' style='color: {"#4a7c59" if pred_30d_return >= 0 else "#8f3f3f"};'>
                {"+" if pred_30d_return >= 0 else ""}{pred_30d_return*100.0:.2f}%
            </div>
            <div style='font-size:0.8rem; color:#6c757d; margin-top:0.4rem;'>SHAP Expected Value Base: <b>{metrics['shap_expected_value']*100.0:+.2f}%</b></div>
        </div>
    """, unsafe_allow_html=True)

    # Display real SHAP waterfall plot
    fig_shap = plot_shap_contributions(shap_vals_array, feature_cols)
    st.pyplot(fig_shap)

# --- PANEL 3: PROJECTED RESPONSE PROFILE (RIGHT) ---
with col_scenario:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Projected Trajectory</h4>", unsafe_allow_html=True)
    
    # Load historical stock prices
    @st.cache_data
    def load_historical_prices(ticker):
        conn = sqlite3.connect(APP_DB_PATH)
        query = """
        SELECT date, close_price 
        FROM stock_prices 
        WHERE ticker = ? 
        ORDER BY date DESC 
        LIMIT 90
        """
        df = pd.read_sql_query(query, conn, params=[ticker])
        conn.close()
        df = df.sort_values('date').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        return df

    hist_df = load_historical_prices(selected_ticker)
    
    # Generate 30-day projection coordinates
    latest_hist_date = hist_df['date'].iloc[-1]
    latest_hist_price = hist_df['close_price'].iloc[-1]
    
    proj_dates = pd.date_range(start=latest_hist_date + pd.Timedelta(days=1), periods=30, freq='D')
    proj_prices = np.linspace(latest_hist_price, target_price, len(proj_dates))
    
    # Build chart frame
    df_plot_hist = pd.DataFrame({
        'Date': hist_df['date'],
        'Historical': hist_df['close_price'],
        'Projected': np.nan
    })
    
    df_plot_proj = pd.DataFrame({
        'Date': proj_dates,
        'Historical': np.nan,
        'Projected': proj_prices
    })
    
    # Connect historical to projection smoothly
    df_plot_proj = pd.concat([
        pd.DataFrame({
            'Date': [latest_hist_date],
            'Historical': [np.nan],
            'Projected': [latest_hist_price]
        }),
        df_plot_proj
    ], ignore_index=True)
    
    chart_df = pd.concat([df_plot_hist, df_plot_proj], ignore_index=True).set_index('Date')
    
    st.line_chart(chart_df, color=["#1a2238", "#c8a84b"], height=200)
    
    # Scenario response metrics table
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#6c757d; text-transform:uppercase; margin-bottom:0.5rem;'>Scenario Impact Metrics</div>", unsafe_allow_html=True)
    
    # Compare scenario against baseline
    baseline_pred = metrics['shap_expected_value']
    scenario_diff = pred_30d_return - baseline_pred
    
    st.markdown(f"""
        <div style='background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px; font-size:0.85rem;'>
            <div class='status-row'><span>Model Average Baseline</span><b>{baseline_pred*100.0:+.2f}%</b></div>
            <div class='status-row'><span>Scenario Return Expectation</span><b style='color: {"#4a7c59" if pred_30d_return >= 0 else "#8f3f3f"};'>{pred_30d_return*100.0:+.2f}%</b></div>
            <div class='status-row'><span>Scenario Net Active Return</span><b style='color: {"#4a7c59" if scenario_diff >= 0 else "#8f3f3f"};'>{scenario_diff*100.0:+.2f}%</b></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#6c757d; text-transform:uppercase; margin-bottom:0.5rem;'>Model Architecture Info</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px; font-size:0.85rem;'>
            <div class='status-row'><span>Framework</span><b>LightGBM Regressor</b></div>
            <div class='status-row'><span>Training Rows</span><b>{metrics['n_training_rows']:,}</b></div>
            <div class='status-row'><span>Total Assets</span><b>{metrics['n_assets']}</b></div>
            <div class='status-row'><span>Features Configured</span><b>{metrics['n_features']}</b></div>
        </div>
    """, unsafe_allow_html=True)