import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page configuration optimized for a clean, professional dashboard view
st.set_page_config(page_title="Macro Factor Impact Explorer", layout="wide")

# ==========================================
# STYLING SHEET: CLEAN LIGHT THEME
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
        padding: 3.5rem 2rem;
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

# Asset target selector configuration
selected_ticker = st.selectbox("Asset Target", ["NVDA", "TSLA", "AAPL", "MSFT"])
st.markdown("---")

# ==========================================
# DATA SIMULATION & MODEL PROCESSING
# ==========================================
@st.cache_resource
def process_macro_model(ticker):
    """Generates sector-specific synthetic matrices and extracts model validation parameters."""
    np.random.seed(hash(ticker) % 10000)
    n_samples = 1500
    
    yield_10y = np.random.uniform(1.5, 6.0, size=n_samples)
    crude_oil = np.random.uniform(40.0, 120.0, size=n_samples)
    usd_idx = np.random.uniform(85.0, 115.0, size=n_samples)
    copper = np.random.uniform(2.5, 5.5, size=n_samples)
    cpi = np.random.uniform(1.0, 10.0, size=n_samples)
    aluminum = np.random.uniform(1000.0, 5000.0, size=n_samples)
    
    if ticker == "NVDA":
        target_return = 4.0 - (yield_10y * 0.4) - (cpi * 0.2) + (usd_idx * 0.05) - (crude_oil * 0.01) + (copper * 0.1) + (aluminum * 0.0001)
    elif ticker == "TSLA":
        target_return = 5.0 - (crude_oil * 0.06) - (yield_10y * 0.5) - (aluminum * 0.0005) - (cpi * 0.3) + (copper * 0.15) + (usd_idx * 0.02)
    elif ticker == "AAPL":
        target_return = 3.5 - (usd_idx * 0.08) - (aluminum * 0.0003) - (copper * 0.4) - (yield_10y * 0.2) - (cpi * 0.1) + (crude_oil * 0.01)
    else: # MSFT
        target_return = 3.0 - (yield_10y * 0.3) - (cpi * 0.1) + (usd_idx * 0.03) - (crude_oil * 0.002) + (copper * 0.01) + (aluminum * 0.00001)

    target_return += np.random.normal(0, 0.8, size=n_samples)
    
    df = pd.DataFrame({
        '10Y Treasury Yield': yield_10y, 'Crude Oil Futures': crude_oil, 'US Dollar Index': usd_idx,
        'Copper Price': copper, 'CPI Inflation Proxy': cpi, 'Aluminum Price': aluminum,
        'Target': target_return
    })
    
    features = ['10Y Treasury Yield', 'Crude Oil Futures', 'US Dollar Index', 'Copper Price', 'CPI Inflation Proxy', 'Aluminum Price']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    test_r2 = rf.score(X_test, y_test)
    global_baseline = np.mean(y_train)
    
    return rf, global_baseline, test_r2

model, model_baseline, model_r2 = process_macro_model(selected_ticker)

# ==========================================
# MAIN INTERACTIVE CORE LAYOUT GRID
# ==========================================
col_inputs, col_model, col_scenario = st.columns([30, 40, 30], gap="large")

# --- PANEL 1: INPUTS (LEFT) ---
with col_inputs:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Inputs</h4>", unsafe_allow_html=True)
    
    sim_yield = st.slider("10Y Treasury Yield (%)", 1.50, 6.00, 4.42, 0.01)
    sim_oil = st.slider("Crude Oil Futures ($/bbl)", 40.00, 120.00, 80.75, 0.25)
    sim_usd = st.slider("US Dollar Index (DXY)", 85.00, 115.00, 99.75, 0.25)
    sim_copper = st.slider("Copper Price ($/lb)", 2.00, 6.00, 4.48, 0.01)
    sim_cpi = st.slider("CPI Inflation Proxy (%)", 1.00, 10.00, 3.20, 0.10)
    sim_aluminum = st.slider("Aluminum Price ($/ton)", 1000.0, 5000.0, 2696.0, 10.0)

# Structural data framing preparation for estimator execution loops
feature_names = ['10Y Treasury Yield', 'Crude Oil Futures', 'US Dollar Index', 'Copper Price', 'CPI Inflation Proxy', 'Aluminum Price']
live_input_df = pd.DataFrame([[sim_yield, sim_oil, sim_usd, sim_copper, sim_cpi, sim_aluminum]], columns=feature_names)

# Compute live model inference prediction
pred_30d_return = model.predict(live_input_df)[0]

ticker_prices = {"NVDA": 205.19, "TSLA": 178.45, "AAPL": 172.50, "MSFT": 415.20}
current_price = ticker_prices[selected_ticker]
target_price = current_price * (1.0 + (pred_30d_return / 100.0))

# --- PANEL 2: MODEL ANALYTICS & RESULTS (CENTER) ---
# One-factor-at-a-time marginal feature sensitivity analysis
feature_contributions = {}
for feature in feature_names:
    isolated_input = pd.DataFrame([[
        sim_yield if feature == '10Y Treasury Yield' else 4.42,
        sim_oil if feature == 'Crude Oil Futures' else 80.75,
        sim_usd if feature == 'US Dollar Index' else 99.75,
        sim_copper if feature == 'Copper Price' else 4.48,
        sim_cpi if feature == 'CPI Inflation Proxy' else 3.20,
        sim_aluminum if feature == 'Aluminum Price' else 2696.0
    ]], columns=feature_names)
    
    feature_contributions[feature] = model.predict(isolated_input)[0] - model_baseline

sorted_drivers = sorted(feature_contributions.items(), key=lambda item: item[1], reverse=True)
biggest_positive_name, biggest_positive_val = sorted_drivers[0]
biggest_negative_name, biggest_negative_val = sorted_drivers[-1]

with col_model:
    st.markdown(f"<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Model Analytics: {selected_ticker}</h4>", unsafe_allow_html=True)
    
    # Corrected vocabulary to acknowledge out-of-sample holdout tracking honestly
    st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #c8a84b;'>
            <div class='metric-label'>Out-of-Sample Test Performance</div>
            <div class='metric-value'>R² = {model_r2:.4f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Current Base Price [{selected_ticker}]</div>
            <div class='metric-value'>${current_price:.2f}</div>
            <div style='font-size:0.8rem; color:#6c757d; margin-top:0.4rem;'>Estimated 30-Day Target: <b>${target_price:.2f}</b></div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Predicted 30-Day Return</div>
            <div class='metric-value' style='color: {"#4a7c59" if pred_30d_return >= 0 else "#8f3f3f"};'>
                {"+" if pred_30d_return >= 0 else ""}{pred_30d_return:.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Cleaned up naming structure to represent the module honestly
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#6c757d; text-transform:uppercase; margin-bottom:0.5rem;'>Factor Sensitivity Ranking (Marginal Contribution Analysis)</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px; font-size:0.85rem;'>
            <div class='status-row'><span>🟢 Biggest Positive Driver</span><span class='text-safe'>{biggest_positive_name} ({biggest_positive_val:+.4f} Δ)</span></div>
            <div class='status-row'><span>🔴 Biggest Negative Driver</span><span class='text-crit'>{biggest_negative_name} ({biggest_negative_val:+.4f} Δ)</span></div>
        </div>
    """, unsafe_allow_html=True)

# --- PANEL 3: PROJECTED RESPONSE PROFILE (RIGHT) ---
with col_scenario:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Projected Response Profile</h4>", unsafe_allow_html=True)
    
    months = np.arange(1, 13, 1)
    
    # Establish flat linear projections
    base_trajectory = np.cumsum(np.repeat(model_baseline * 0.05, 12))
    scenario_trajectory = np.cumsum(np.repeat(pred_30d_return * 0.05, 12))
    
    chart_df = pd.DataFrame({
        'Timeline (Months)': months,
        'Base Expectation Profile (%)': base_trajectory,
        'User Scenario Response (%)': scenario_trajectory
    }).set_index('Timeline (Months)')
    
    st.line_chart(chart_df, color=["#6c757d", "#c8a84b"], height=190)
    
    st.markdown(f"""
        <div class='status-row' style='margin-top:0.5rem;'><span>Base Cumulative Expectation</span><b>{base_trajectory[-1]:+.2f}%</b></div>
        <div class='status-row'><span>Scenario Cumulative Response</span><span style='color: {"#4a7c59" if scenario_trajectory[-1] >= base_trajectory[-1] else "#8f3f3f"};'><b>{scenario_trajectory[-1]:+.2f}%</b></span></div>
    """, unsafe_allow_html=True)