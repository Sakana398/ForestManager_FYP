# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *

st.set_page_config(page_title="ForestManager | 3D Map", layout="wide")

st.title("🗺️ 3D Forest Structure Visualization")

# --- CSS FOR LEGEND ---
st.markdown("""
    <style>
    .map-legend {
        position: fixed; bottom: 30px; right: 30px; background-color: rgba(255, 255, 255, 0.90); 
        color: #333; padding: 15px; border-radius: 8px; border: 1px solid #ccc; z-index: 9999;
        font-family: 'Segoe UI', sans-serif; font-size: 13px; box-shadow: 0px 4px 15px rgba(0,0,0,0.2); 
        min-width: 160px; backdrop-filter: blur(4px);
    }
    .legend-title { font-weight: 700; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 5px; display: block; font-size: 14px;}
    .legend-item { display: flex; align-items: center; margin-bottom: 6px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; display: inline-block; border: 1px solid #999;}
    </style>
""", unsafe_allow_html=True)

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    
    # 1. CLEAN DATA
    for col in [COL_X, COL_Y, COL_CURRENT]:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)
    
    # Filter out trees with invalid coordinates (0,0)
    df_all = df_all[(df_all[COL_X] > 100) & (df_all[COL_Y] > 0)]
    
    # -----------------------------------------------------------
    # 🌍 CRITICAL FIX: USE RAW COORDINATES DIRECTLY
    # -----------------------------------------------------------
    # Your XCO/YCO are already Longitude/Latitude
    df_all['lon_viz'] = df_all[COL_X]
    df_all['lat_viz'] = df_all[COL_Y]
    
    # Pack into a single list column for PyDeck: [Lon, Lat]
    df_all['trunk_coords'] = df_all[['lon_viz', 'lat_viz']].values.tolist()

    # Calculate Z (Height) for the Crown
    # Visual Scale: Height (m) = DBH (cm) * 1.0
    df_all['tree_height'] = df_all[COL_CURRENT] * 1.0
    
    # Pack Crown Coords: [Lon, Lat, Height]
    df_all['crown_coords'] = df_all[['lon_viz', 'lat_viz', 'tree_height']].values.tolist()
    
    # Crown Radius (Visual)
    df_all['crown_radius'] = (df_all[COL_CURRENT] / 100.0) * 5.0 
    df_all['crown_radius'] = df_all['crown_radius'].clip(lower=1.0, upper=8.0)

    # Tooltips & Colors
    if 'Mortality_Risk' not in df_all.columns: df_all['Mortality_Risk'] = 0.0
    df_all['Mortality_Risk'] = df_all['Mortality_Risk'].fillna(0.0)
    df_all['Tooltip_Risk'] = (df_all['Mortality_Risk'] * 100).round(1).astype(str) + '%'

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
            view_mode = st.radio("Color Mode:", ["Thinning Candidates (Action)", "Mortality Risk (Heatmap)", "Post-Thinning Scenario"], horizontal=True)
        with col_c2:
            min_dbh_view = st.slider("Hide Small Trees (< cm):", 0, 50, 10, 5)
        with col_c3:
            opacity = st.slider("Canopy Opacity:", 0.1, 1.0, 0.8, 0.1)

    # ==========================================
    # 3. COLOR LOGIC
    # ==========================================
    df_view = df_all[df_all[COL_CURRENT] >= min_dbh_view].copy()
    
    def get_crown_color(row):
        try:
            if "Mortality" in view_mode:
                risk = max(0.0, min(1.0, float(row['Mortality_Risk'])))
                r = int(255 * risk)
                g = int(255 * (1 - risk))
                return [r, g, 50, int(opacity * 255)]
            elif "Thinning" in view_mode:
                if row[COL_ID] in thinning_tags:
                    return [255, 80, 80, int(opacity * 255)] # Coral
                else:
                    return [60, 180, 130, int(opacity * 255)] # Green
            else: 
                return [60, 180, 130, int(opacity * 255)]
        except:
            return [128, 128, 128, 150]

    if "Post-Thinning" in view_mode:
        df_view = df_view[~df_view[COL_ID].isin(thinning_tags)]

    # ==========================================
    # 4. RENDER MAP
    # ==========================================
    if not df_view.empty:
        df_view['crown_color'] = df_view.apply(get_crown_color, axis=1)
        
        # LAYER 1: TRUNK (Wood Cylinder)
        layer_trunk = pdk.Layer(
            "ColumnLayer",
            data=df_view,
            get_position='trunk_coords',
            get_elevation='tree_height',
            elevation_scale=1.0,
            radius=0.3,                   
            get_fill_color=[101, 67, 33], # Brown
            pickable=False,
            extruded=True,
        )

        # LAYER 2: CROWN (Leafy Sphere)
        layer_crown = pdk.Layer(
            "ScatterplotLayer",
            data=df_view,
            get_position='crown_coords',  # <--- [Lon, Lat, Height]
            get_radius="crown_radius",
            radius_scale=1,
            get_fill_color="crown_color",
            pickable=True,
            stroked=False,
            filled=True,
        )

        view_state = pdk.ViewState(
            longitude=df_view['lon_viz'].mean(),
            latitude=df_view['lat_viz'].mean(),
            zoom=18,
            pitch=60,
            bearing=0
        )

        tooltip_html = (
            f"<b>ID:</b> {{{COL_ID}}}<br>"
            f"<b>Species:</b> {{{COL_SPECIES}}}<br>"
            f"<b>Size:</b> {{{COL_CURRENT}}} cm<br>"
            f"<b>Risk:</b> {{Tooltip_Risk}}"
        )

        # ------------------------------------------------
        # 🔑 INTEGRATED MAPBOX KEY HERE
        # ------------------------------------------------
        try:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/satellite-streets-v11", # Adds the "Plains" / Satellite view
                layers=[layer_trunk, layer_crown],
                initial_view_state=view_state, 
                tooltip={"html": tooltip_html, "style": {"backgroundColor": "#222", "color": "#fff", "zIndex": "9999"}},
                api_keys={"mapbox": st.secrets["mapbox"]["token"]} # Loads key from secrets.toml
            ))
        except Exception as e:
            st.error(f"⚠️ Map Loading Error: {e}. Please check your .streamlit/secrets.toml file.")

        # LEGEND
        if "Mortality" in view_mode:
            legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Mortality Risk</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF5050;'></span>High Risk (Dying)</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #FFDD44;'></span>Moderate Risk</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #44CC44;'></span>Healthy</div>
                </div>
            """
        elif "Thinning" in view_mode:
             legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Action Plan</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #FF5050;'></span>Removal Candidate</div>
                    <div class='legend-item'><span class='legend-dot' style='background: #3CB482;'></span>Crop Tree (Keep)</div>
                </div>
            """
        else:
             legend_html = """
                <div class='map-legend'>
                    <span class='legend-title'>Post-Thinning</span>
                    <div class='legend-item'><span class='legend-dot' style='background: #3CB482;'></span>Remaining Stock</div>
                </div>
            """
        st.markdown(legend_html, unsafe_allow_html=True)

    else:
        st.warning("No trees visible. Try adjusting the filter.")
    
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Dashboard"): st.switch_page("pages/0_Dashboard.py")
    with col_n2:
        if st.button("Go to Growth Trends ➡️"): st.switch_page("pages/2_Individual_Growth.py")

else:
    st.warning("⚠️ Data not loaded.")