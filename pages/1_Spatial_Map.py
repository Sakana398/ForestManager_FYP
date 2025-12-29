# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *

st.set_page_config(page_title="ForestManager | 3D Map", layout="wide")

st.title("🗺️ 3D Forest Structure Visualization")

# --- CLEANER LEGEND STYLE ---
st.markdown("""
    <style>
    .map-legend {
        position: fixed; bottom: 30px; right: 30px; background-color: rgba(255, 255, 255, 0.95); 
        color: #333; padding: 15px; border-radius: 8px; border: 1px solid #ddd; z-index: 9999;
        font-family: 'Segoe UI', sans-serif; font-size: 13px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); 
        min-width: 160px; backdrop-filter: blur(5px);
    }
    .legend-title { font-weight: 600; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 5px; display: block; font-size: 14px;}
    .legend-item { display: flex; align-items: center; margin-bottom: 6px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    
    # Cleaning
    for col in [COL_X, COL_Y, COL_CURRENT]:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)
    
    if 'Mortality_Risk' not in df_all.columns: df_all['Mortality_Risk'] = 0.0
    df_all['Mortality_Risk'] = df_all['Mortality_Risk'].fillna(0.0)
    
    # Safe Tooltip String
    df_all['Tooltip_Risk'] = (df_all['Mortality_Risk'] * 100).round(1).astype(str) + '%'
    df_all = df_all.dropna(subset=[COL_X, COL_Y])

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
            # Increased default filter to 10cm for cleaner initial view
            min_dbh_view = st.slider("Hide Small Trees (< cm):", 0, 50, 10, 5)

        with col_c3:
            # Default trunk radius reduced to 0.15 for realism
            radius_size = st.slider("Trunk Thickness (m):", 0.05, 2.0, 0.15, 0.05)

    # ==========================================
    # 3. COLOR LOGIC (CLEANER PALETTE)
    # ==========================================
    df_view = df_all[df_all[COL_CURRENT] >= min_dbh_view].copy()
    max_h = df_view[COL_CURRENT].max() if not df_view.empty else 1.0
    
    def get_color(row):
        try:
            # MODE A: MORTALITY HEATMAP
            if "Mortality" in view_mode:
                risk = max(0.0, min(1.0, float(row['Mortality_Risk'])))
                # Gradient: Green (Low Risk) -> Yellow -> Red (High Risk)
                # Using smoother RGB values
                r = int(255 * risk)
                g = int(255 * (1 - risk))
                return [r, g, 50, 200]

            # MODE B: THINNING CANDIDATES
            elif "Thinning" in view_mode:
                if row[COL_ID] in thinning_tags:
                    return [255, 100, 100, 220] # Soft Red (Coral)
                else:
                    return [50, 180, 160, 180]  # Teal/Sea Green
            
            # MODE C: POST-THINNING
            else: 
                return [50, 180, 160, 180]      # Teal/Sea Green
        except:
            return [128, 128, 128, 150]

    # Filter Post-Thinning
    if "Post-Thinning" in view_mode:
        df_view = df_view[~df_view[COL_ID].isin(thinning_tags)]

    # ==========================================
    # 4. RENDER MAP
    # ==========================================
    if not df_view.empty:
        df_view['color'] = df_view.apply(get_color, axis=1)

        layer = pdk.Layer(
            "ColumnLayer",
            data=df_view,
            get_position=[COL_X, COL_Y],
            get_elevation=COL_CURRENT,
            elevation_scale=1.5,
            radius=radius_size, 
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        view_state = pdk.ViewState(
            longitude=df_view[COL_X].mean(),
            latitude=df_view[COL_Y].mean(),
            zoom=16, pitch=50
        )

        tooltip_html = (
            f"<b>ID:</b> {{{COL_ID}}}<br>"
            f"<b>Species:</b> {{{COL_SPECIES}}}<br>"
            f"<b>Size:</b> {{{COL_CURRENT}}} cm<br>"
            f"<b>Risk:</b> {{Tooltip_Risk}}"
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10", 
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip={"html": tooltip_html, "style": {"backgroundColor": "#222", "color": "#fff", "zIndex": "9999"}}
        ))

        # DYNAMIC LEGEND
        if "Mortality" in view_mode:
            legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Mortality Risk</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF4444;'></span>High Risk</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #FFDD44;'></span>Moderate Risk</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #44CC44;'></span>Healthy</div>
                </div>
            """
        elif "Thinning" in view_mode:
             legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Action Plan</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF6464;'></span>Removal Candidate</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #32B4A0;'></span>Crop Tree (Keep)</div>
                </div>
            """
        else:
             legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Post-Thinning</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #32B4A0;'></span>Remaining Stock</div>
                </div>
            """
            
        st.markdown(legend_html, unsafe_allow_html=True)
    else:
        st.warning("No trees visible.")

    # Navigation
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Dashboard"): st.switch_page("pages/0_Dashboard.py")
    with col_n2:
        if st.button("Go to Growth Trends ➡️"): st.switch_page("pages/2_Individual_Growth.py")