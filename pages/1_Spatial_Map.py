# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *

st.set_page_config(page_title="ForestManager | 3D Map", layout="wide")

st.title("🗺️ 3D Forest Structure Visualization")

# --- IMPROVED CSS FOR READABLE LEGEND ---
st.markdown("""
    <style>
    .map-legend {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #ffffff; /* Solid White */
        color: #333333;            /* Dark Grey Text (Forced) */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        z-index: 9999;             /* Ensure it is on top */
        font-family: sans-serif;
        font-size: 14px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2); /* Strong Shadow */
        min-width: 180px;
    }
    .legend-title {
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 15px;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
        display: block;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        display: inline-block;
        border: 1px solid rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    
    # 1. CLEAN & PREPARE DATA
    df_all[COL_X] = pd.to_numeric(df_all[COL_X], errors='coerce')
    df_all[COL_Y] = pd.to_numeric(df_all[COL_Y], errors='coerce')
    df_all[COL_CURRENT] = pd.to_numeric(df_all[COL_CURRENT], errors='coerce').fillna(0)
    df_all = df_all.dropna(subset=[COL_X, COL_Y])

    if not st.session_state['df_thinning_recs'].empty:
        thinning_tags = set(st.session_state['df_thinning_recs'][COL_ID])
    else:
        thinning_tags = set()

    # ==========================================
    # 2. CONTROLS
    # ==========================================
    with st.expander("🛠️ Map Settings & Layers", expanded=True):
        col_c1, col_c2, col_c3, col_c4 = st.columns([1.5, 1, 1, 1])
        
        with col_c1:
            view_mode = st.radio(
                "Scenario View:", 
                ["Current Forest (Highlight Targets)", "Post-Thinning (Remaining Only)"], 
                horizontal=True
            )
        
        with col_c2:
            map_style = st.selectbox(
                "Base Map Style:",
                ["mapbox://styles/mapbox/light-v10", "mapbox://styles/mapbox/satellite-v9", "mapbox://styles/mapbox/dark-v10"],
                index=0,
                format_func=lambda x: "Light Map" if "light" in x else ("Satellite" if "satellite" in x else "Dark Map")
            )

        with col_c3:
            elevation_scale = st.slider("Height Exaggeration:", 0.1, 5.0, 1.5, 0.1)
            
        with col_c4:
            radius_size = st.slider("Canopy/Stem Radius:", 0.1, 5.0, 2.0, 0.1)

    # ==========================================
    # 3. COLOR LOGIC
    # ==========================================
    max_h = df_all[COL_CURRENT].max()
    
    def get_tree_color(row):
        # 1. Check if tree is marked for removal
        if row[COL_ID] in thinning_tags:
            return [220, 20, 60, 255] # 🔴 Bright Red
            
        # 2. If keeping, calculate green intensity based on height
        # Normalize height 0 to 1
        h_norm = min(row[COL_CURRENT] / max_h, 1.0) if max_h > 0 else 0.5
        
        # Gradient Green
        r = int(100 - (80 * h_norm))
        g = int(220 - (120 * h_norm))
        b = int(100 - (70 * h_norm))
        
        return [r, g, b, 200]

    # Filter data based on view mode
    if "Post-Thinning" in view_mode:
        plot_data = df_all[~df_all[COL_ID].isin(thinning_tags)].copy()
        
        legend_html = """
            <div class='map-legend'>
                <span class='legend-title'>Post-Thinning Status</span>
                <div class='legend-item'>
                    <span class='legend-dot' style='background: #228B22;'></span>
                    <span>Remaining Trees</span>
                </div>
                <div style='margin-top:5px; font-size:12px; color:#666;'>
                    <i>Darker Green = Taller Tree</i>
                </div>
            </div>
        """
    else:
        plot_data = df_all.copy()
        count_remove = len(thinning_tags)
        
        legend_html = f"""
            <div class='map-legend'>
                <span class='legend-title'>Current Status</span>
                <div class='legend-item'>
                    <span class='legend-dot' style='background: #DC143C;'></span>
                    <span>Removal Candidates ({count_remove})</span>
                </div>
                <div class='legend-item'>
                    <span class='legend-dot' style='background: #90EE90;'></span>
                    <span>Crop Trees (Keep)</span>
                </div>
                <div style='margin-top:5px; font-size:12px; color:#666;'>
                    <i>Darker Green = Taller Tree</i>
                </div>
            </div>
        """

    # Apply Color Function
    plot_data['color'] = plot_data.apply(get_tree_color, axis=1)

    # ==========================================
    # 4. RENDER MAP
    # ==========================================
    if not plot_data.empty:
        mid_x = plot_data[COL_X].mean()
        mid_y = plot_data[COL_Y].mean()
        
        layer = pdk.Layer(
            "ColumnLayer",
            data=plot_data,
            get_position=[COL_X, COL_Y],
            get_elevation=COL_CURRENT,
            elevation_scale=elevation_scale,
            radius=radius_size,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        view_state = pdk.ViewState(
            longitude=mid_x,
            latitude=mid_y,
            zoom=16,
            pitch=50,
            bearing=0
        )

        # Tooltip
        tooltip_html = f"""
            <div style='font-family: sans-serif; padding: 5px; font-size: 13px;'>
                <b>ID:</b> {{{COL_ID}}}<br>
                <b>Species:</b> {{{COL_SPECIES}}}<br>
                <b>Group:</b> {{{COL_SPECIES_GRP}}}<br>
                <b>Size ({COL_CURRENT}):</b> {{{COL_CURRENT}}} cm
            </div>
        """
        
        tooltip = {
            "html": tooltip_html,
            "style": {
                "backgroundColor": "#333", 
                "color": "white", 
                "borderRadius": "4px",
                "border": "1px solid white",
                "zIndex": "10000"
            }
        }

        r = pdk.Deck(
            map_style=map_style,
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
        )
        
        # Render Map
        st.pydeck_chart(r)
        
        # Render Legend Overlay
        st.markdown(legend_html, unsafe_allow_html=True)
        
    else:
        st.error("Error: No data available to display.")

    # ==========================================
    # 5. NAVIGATION
    # ==========================================
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Dashboard"): st.switch_page("pages/0_Dashboard.py")
    with col_n2:
        if st.button("Go to Growth Trends ➡️"): st.switch_page("pages/2_Individual_Growth.py")

else:
    st.warning("⚠️ Data not loaded. Please go to the **Home** page first.")