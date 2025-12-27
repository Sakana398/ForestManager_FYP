# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *

st.set_page_config(page_title="ForestManager | 3D Map", layout="wide")

st.title("🗺️ 3D Forest Structure Visualization")

# --- LEGEND CSS ---
st.markdown("""
    <style>
    .map-legend {
        position: fixed; bottom: 30px; right: 30px; background-color: #ffffff; color: #333;
        padding: 15px; border-radius: 8px; border: 1px solid #ccc; z-index: 9999;
        font-family: sans-serif; font-size: 14px; box-shadow: 0px 4px 12px rgba(0,0,0,0.2); min-width: 180px;
    }
    .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
    .legend-dot { width: 14px; height: 14px; border-radius: 50%; margin-right: 10px; display: inline-block; border: 1px solid #ddd;}
    .legend-title { font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 5px; display: block;}
    </style>
""", unsafe_allow_html=True)

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    
    # ==========================================
    # 1. DATA CLEANING & SAFETY
    # ==========================================
    df_all[COL_X] = pd.to_numeric(df_all[COL_X], errors='coerce')
    df_all[COL_Y] = pd.to_numeric(df_all[COL_Y], errors='coerce')
    df_all[COL_CURRENT] = pd.to_numeric(df_all[COL_CURRENT], errors='coerce').fillna(0)
    
    # Fix Mortality Risk (Handle Missing Columns or NaNs)
    if 'Mortality_Risk' not in df_all.columns:
        df_all['Mortality_Risk'] = 0.0
    
    # Fill NaNs with 0.0 to prevent "ValueError: cannot convert float NaN to integer"
    df_all['Mortality_Risk'] = df_all['Mortality_Risk'].fillna(0.0)
    
    # Create a nice string for the tooltip (e.g., "12.5%")
    df_all['Tooltip_Risk'] = (df_all['Mortality_Risk'] * 100).round(1).astype(str) + '%'

    # Drop rows with bad coordinates
    df_all = df_all.dropna(subset=[COL_X, COL_Y])

    # Get thinning tags
    if not st.session_state['df_thinning_recs'].empty:
        thinning_tags = set(st.session_state['df_thinning_recs'][COL_ID])
    else:
        thinning_tags = set()

    # ==========================================
    # 2. CONTROLS
    # ==========================================
    with st.expander("🛠️ Map Settings", expanded=True):
        col_c1, col_c2, col_c3 = st.columns([1.5, 1, 1])
        
        with col_c1:
            view_mode = st.radio(
                "Color Mode:", 
                ["Thinning Candidates (Action)", "Mortality Risk (Heatmap)", "Post-Thinning Scenario"], 
                horizontal=True
            )
        
        with col_c2:
            min_dbh_view = st.slider("Hide Small Trees (< cm):", 0, 50, 5, 5)

        with col_c3:
            elevation_scale = st.slider("Height Scale:", 0.1, 5.0, 1.5, 0.1)

    # ==========================================
    # 3. COLOR LOGIC
    # ==========================================
    # Filter by size
    df_view = df_all[df_all[COL_CURRENT] >= min_dbh_view].copy()
    max_h = df_view[COL_CURRENT].max() if not df_view.empty else 1.0
    
    def get_color(row):
        try:
            # MODE A: MORTALITY HEATMAP
            if "Mortality" in view_mode:
                risk = row['Mortality_Risk'] # 0.0 to 1.0
                # Clamp risk between 0 and 1 just in case
                risk = max(0.0, min(1.0, risk))
                
                # Gradient: Low(Green) -> High(Red)
                r = int(255 * risk)
                g = int(255 * (1 - risk))
                b = 0
                return [r, g, b, 180]

            # MODE B: THINNING CANDIDATES
            elif "Thinning" in view_mode:
                if row[COL_ID] in thinning_tags:
                    return [255, 0, 0, 200] # Red
                else:
                    # Height-based Green
                    h_norm = min(row[COL_CURRENT] / max_h, 1.0)
                    g = int(100 + (100 * h_norm)) # Range 100-200
                    return [50, g, 50, 140]
            
            # MODE C: POST-THINNING
            else: 
                h_norm = min(row[COL_CURRENT] / max_h, 1.0)
                g = int(100 + (100 * h_norm))
                return [50, g, 50, 140]
        except:
            return [100, 100, 100, 100] # Grey fallback on error

    # Apply Filters for Post-Thinning View
    if "Post-Thinning" in view_mode:
        df_view = df_view[~df_view[COL_ID].isin(thinning_tags)]

    # ==========================================
    # 4. RENDER MAP
    # ==========================================
    if not df_view.empty:
        # Calculate colors safely
        df_view['color'] = df_view.apply(get_color, axis=1)

        layer = pdk.Layer(
            "ColumnLayer",
            data=df_view,
            get_position=[COL_X, COL_Y],
            get_elevation=COL_CURRENT,
            elevation_scale=elevation_scale,
            radius=0.4, 
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        view_state = pdk.ViewState(
            longitude=df_view[COL_X].mean(),
            latitude=df_view[COL_Y].mean(),
            zoom=16, 
            pitch=50
        )

        # Simplified Tooltip using the pre-calculated string
        tooltip = {
            "html": f"<b>ID:</b> {{{COL_ID}}}<br>"
                    f"<b>Species:</b> {{{COL_SPECIES}}}<br>"
                    f"<b>Size:</b> {{{COL_CURRENT}}} cm<br>"
                    f"<b>Risk:</b> {{Tooltip_Risk}}",
            "style": {
                "backgroundColor": "#333", 
                "color": "white", 
                "borderRadius": "4px",
                "padding": "8px",
                "zIndex": "10000"
            }
        }

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10", 
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip=tooltip
        ))

        # DYNAMIC LEGEND
        if "Mortality" in view_mode:
            legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Mortality Risk</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF0000;'></span>High Risk (>80%)</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #FFFF00;'></span>Moderate Risk</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #00FF00;'></span>Low Risk (Healthy)</div>
                </div>
            """
        elif "Thinning" in view_mode:
             legend_html = f"""
                <div class='map-legend'>
                    <span class='legend-title'>Action Plan</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF0000;'></span>Removal Candidate</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #50C850;'></span>Crop Tree (Keep)</div>
                </div>
            """
        else:
             legend_html = f"""
                <div class='map-legend'>
                    <span class='legend-title'>Post-Thinning</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #50C850;'></span>Remaining Stock</div>
                </div>
            """
            
        st.markdown(legend_html, unsafe_allow_html=True)
        
    else:
        st.warning("No trees visible. Try adjusting the 'Hide Small Trees' slider.")

    # Navigation
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Dashboard"): st.switch_page("pages/0_Dashboard.py")
    with col_n2:
        if st.button("Go to Growth Trends ➡️"): st.switch_page("pages/2_Individual_Growth.py")

else:
    st.warning("⚠️ Data not loaded. Please go to the **Home** page first.")