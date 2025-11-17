# src/components.py
import streamlit as st
from .config import *

def render_sidebar_filters(df):
    st.sidebar.header("Dashboard Filters")
    
    # Reset Button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.session_state.clear()
        st.rerun()
    
    if 'QUAD' in df.columns:
        all_quads = sorted(df['QUAD'].dropna().unique().astype(int))
        sel_quads = st.sidebar.multiselect("Select Quadrant(s):", all_quads, default=all_quads)
    else:
        sel_quads = []

    if 'SP' in df.columns:
        all_species = sorted(df['SP'].dropna().unique())
        st.sidebar.caption(f"{len(all_species)} Species Found")
        sel_species = st.sidebar.multiselect("Select Species:", all_species, default=all_species[:5])
    else:
        sel_species = []
        
    return sel_quads, sel_species

def render_thinning_controls(df):
    st.sidebar.header("Thinning Simulation")
    
    growth_pct = st.sidebar.slider(
        "1. Low Growth Rate (% Below)", 0, 100, DEFAULT_GROWTH_PERCENTILE
    )
    
    max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
    ci_limit = st.sidebar.slider(
        "2. High Competition (CI ≥)", 0, max_ci, 0,
        help="Hegyi's Index. Higher value = More pressure from neighbors."
    )
    
    prox_lim = st.sidebar.slider(
        "3. Close Proximity (Distance ≤)", 0.0, 10.0, DEFAULT_PROXIMITY
    )
    
    return growth_pct, ci_limit, prox_lim