import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Config configured for full-width structural layout
st.set_page_config(page_title="Macroeconomic AI Research Engine", layout="wide")

# ==========================================
# CUSTOM CSS: LIGHT CRUNCH THEME WITH FIXES
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
    
    /* Clean up standard Streamlit layout borders */
    span[data-testid="stSidebarCollapseButton"], 
    div[data-testid="collapsedControl"],
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Base Application Background */
    .stApp {
        background-color: #ffffff;
        color: #2d2d2d;
        font-family: 'Inter', sans-serif;
    }
    
    /* Force all slider labels to deep slate gray */
    label, [data-testid="stWidgetLabel"] p {
        color: #1a2238 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* Standard font overrides */
    h1, h2, h3, h4, .mono-text {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Full-width Top Banner */
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
    
    /* Structured component cards */
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
    
    /* Plain horizontal metrics progress bars */
    .bar-wrapper {
        width: 100%;
        height: 6px;
        background-color: #e9ecef;
        margin-top: 0.5rem;
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background-color: #c8a84b;
        transition: width 0.4s ease;
    }
    
    /* Diagnostics HUD row layouts */
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #e9ecef;
        font-size: 0.85rem;
        color: #2d2d2d;
    }
    .text-safe { color: #4a7c59; font-weight: 600; }
    .text-warn { color: #a07c3a; font-weight: 600; }
    .text-crit { color: #8f3f3f; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MULTI-VARIABLE MACRO INTELLIGENCE CORE
# ==========================================
@st.cache_resource
def initialize_macro_model_pipeline():
    """Generates synthetic macroeconomic core historical metrics arrays and fits estimators."""
    np.random.seed(101)
    n_samples = 1200
    
    # Simulating standard macroeconomic feature nodes
    interest_rate = np.random.uniform(0.25, 6.5, size=n_samples)
    cpi_inflation = np.random.uniform(1.0, 9.0, size=n_samples)
    unemployment = np.random.uniform(3.0, 10.0, size=n_samples)
    
    # Structural financial target generation functions
    gdp_growth = 4.0 - (interest_rate * 0.3) - (cpi_inflation * 0.15) + np.random.normal(0, 0.4, size=n_samples)
    bond_yield_spread = 0.5 + (interest_rate * 0.25) + (cpi_inflation * 0.1) + np.random.normal(0, 0.15, size=n_samples)
    consumer_spending = 5.0 - (unemployment * 0.4) - (interest_rate * 0.1) + np.random.normal(0, 0.3, size=n_samples)
    market_volatility = 12.0 + (unemployment * 1.5) + (cpi_inflation * 0.8) + np.random.normal(0, 1.5, size=n_samples)

    data_payload = pd.DataFrame({
        'rates': interest_rate, 'cpi': cpi_inflation, 'unemp': unemployment,
        'gdp': gdp_growth, 'spread': bond_yield_spread, 'consumer': consumer_spending, 'vix': market_volatility
    })
    
    features = ['rates', 'cpi', 'unemp']
    models_dict = {}
    
    # Train separate predictive matrices for each targeted financial indicator
    for target in ['gdp', 'spread', 'consumer', 'vix']:
        X = data_payload[features]
        y = data_payload[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models_dict[target] = rf
        
    return models_dict

models = initialize_macro_model_pipeline()

# ==========================================
# VISUAL HERO HEADER BANNER
# ==========================================
st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>MACROECONOMIC AI RESEARCH ENGINE</div>
        <div class='hero-subtitle'>Multi-variable machine learning forecasting framework predicting asset volatility and growth indicators from historical indices.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# THREE COLUMN COMPONENT MAIN LAYOUT
# ==========================================
col_input, col_metrics, col_hud = st.columns([25, 45, 30], gap="large")

# COLUMN 1: RESEARCH PARAMETERS CONTROL PANEL (LEFT)
with col_input:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Research Inputs</h4>", unsafe_allow_html=True)
    input_rates = st.slider("Federal Funds Rate (%)", 0.25, 7.00, 3.50, 0.25)
    input_cpi = st.slider("Consumer Price Index (CPI Inflation %)", 1.0, 10.0, 2.8, 0.1)
    input_unemp = st.slider("Unemployment Rate (%)", 2.5, 11.0, 4.2, 0.1)

# Compute real-time macroeconomic inference vectors
eval_vector = [[input_rates, input_cpi, input_unemp]]
pred_gdp = models['gdp'].predict(eval_vector)[0]
pred_spread = models['spread'].predict(eval_vector)[0]
pred_cons = models['consumer'].predict(eval_vector)[0]
pred_vix = models['vix'].predict(eval_vector)[0]

# COLUMN 2: FORWARD ESTIMATES GRID (CENTER)
with col_metrics:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Forward Estimates</h4>", unsafe_allow_html=True)
    
    # Block 1: Annualized GDP Forecast
    fill_gdp = min(100, max(5, int(((pred_gdp + 2) / 7) * 100)))  # Scale accounting for potential minor negative shifts
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Annualized GDP Growth Estimate</div>
            <div class='metric-value'>{pred_gdp:.2f}%</div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_gdp}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Block 2: Corporate Bond Yield Spread
    fill_spread = min(100, max(5, int((pred_spread / 4.5) * 100)))
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Corporate Bond Yield Spread</div>
            <div class='metric-value'>{pred_spread:.3f} <span style='font-size: 14px; color:#6c757d;'>bps</span></div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_spread}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Block 3: Consumer Spending Growth index
    fill_cons = min(100, max(5, int((pred_cons / 6.0) * 100)))
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Consumer Spending Momentum Index</div>
            <div class='metric-value'>{pred_cons:.2f}</div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_cons}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)

# COLUMN 3: OVERALL SYSTEM STATUS & SPECS (RIGHT)
with col_hud:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Market Stability Risk Matrix</h4>", unsafe_allow_html=True)
    
    gdp_status = ("Contraction Risk", "text-crit") if pred_gdp < 0.5 else (("Stagnant", "text-warn") if pred_gdp < 1.8 else ("Expansionary", "text-safe"))
    vix_status = ("High Volatility", "text-crit") if pred_vix > 24.0 else (("Moderate Stress", "text-warn") if pred_vix > 15.0 else ("Stable Equities", "text-safe"))
    inf_status = ("Hyper-Inflationary", "text-crit") if input_cpi > 5.5 else (("Elevated Strain", "text-warn") if input_cpi > 3.0 else ("Target Anchored", "text-safe"))
    
    st.markdown(f"""
        <div style='background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px;'>
            <div class='status-row'><span>Economic Cycle Vector</span><span class='{gdp_status[1]}'>{gdp_status[0]}</span></div>
            <div class='status-row'><span>Equity Implied Volatility (VIX)</span><span class='{vix_status[1]}'>{pred_vix:.1f}</span></div>
            <div class='status-row'><span>Monetary Stability Bounds</span><span class='{inf_status[1]}'>{inf_status[0]}</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    # 12-Month Treasury Curve Term Premium decay simulation representation plot
    st.markdown("<br><div style='font-size:0.75rem; font-weight:600; color:#6c757d; text-transform:uppercase;'>Simulated 12-Month Treasury Yield Shift Curve</div>", unsafe_allow_html=True)
    months = np.arange(1, 13, 1)
    
    # Dynamic yield calculations mapping flattening vs steepening inverted yield curve profiles
    base_yield = input_rates + (pred_spread * 0.2)
    decay_curve = base_yield - ((input_cpi * 0.04) * months)
    
    chart_data = pd.DataFrame({
        'Timeline (Months)': months, 
        'Projected Treasury Yield (%)': decay_curve
    }).set_index('Timeline (Months)')
    st.line_chart(chart_data, height=160)