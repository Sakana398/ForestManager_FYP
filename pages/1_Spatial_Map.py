# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *
from src.utils import standardize_coordinates

st.set_page_config(page_title="ForestManager | Digital Twin", layout="wide")
st.title("🗺️ Forest Digital Twin")

# ==========================================
# 1. MAPBOX KEY SETUP (Simplified)
# ==========================================
map_provider = "mapbox"
map_style = "mapbox://styles/mapbox/satellite-v9"
mapbox_key = None

# OPTION A: Hardcode for testing (Uncomment if secrets fail)
# mapbox_key = "pk.eyJ1Ijo..." 

# OPTION B: Load from Secrets (Flat Key Only)
if not mapbox_key:
    if "MAPBOX_KEY" in st.secrets:
        mapbox_key = st.secrets["MAPBOX_KEY"]

# Validation
if not mapbox_key:
    # Fail-safe: Switch to CartoDB if key is missing
    map_provider = "carto"
    map_style = "dark"
    st.toast("⚠️ Mapbox Key missing. Switched to Backup Map.", icon="🗺️")

# ==========================================
# 2. DATA PREP
# ==========================================
if 'df' in st.session_state:
    raw_df = st.session_state['df'].copy()
    
    # 🧹 USE UTILITY FUNCTION (Fixes Coordinates Automatically)
    # This ensures trees land in Pasoh, not the ocean.
    raw_df = standardize_coordinates(raw_df)

    # Remove rows that would crash the map
    raw_df = raw_df.dropna(subset=['lon', 'lat', COL_ID])

    # Load Thinning Strategy
    thinning_ids = set()
    if 'df_thinning_recs' in st.session_state and not st.session_state['df_thinning_recs'].empty:
         thinning_ids = set(st.session_state['df_thinning_recs'][COL_ID].astype(str))

    # ==========================================
    # 3. FILTERS (Top Row Layout)
    # ==========================================
    c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
    
    with c1:
        st.write("#### 🎯 Quick Find")
        if thinning_ids:
            if st.button(f"✨ Find Candidate ({len(thinning_ids)})", type="primary", use_container_width=True):
                candidates = raw_df[raw_df[COL_ID].astype(str).isin(thinning_ids)]
                if not candidates.empty:
                    st.session_state['map_search_tag'] = candidates.sample(1).iloc[0][COL_ID]
                    st.rerun()
                else:
                    st.warning("No candidates found in current view.")
        else:
            st.button("✨ Find Tree...", disabled=True, use_container_width=True)

    with c2:
        st.write("#### 🌿 Filter Species")
        all_species = sorted(raw_df[COL_SPECIES].unique()) if COL_SPECIES in raw_df.columns else []
        sel_species = st.selectbox("Species", ["All"] + all_species, label_visibility="collapsed")

    with c3:
        st.write("#### 🏷️ Select Tree Tag")
        if sel_species != "All":
            filtered_df = raw_df[raw_df[COL_SPECIES] == sel_species]
        else:
            filtered_df = raw_df
            
        available_tags = sorted(filtered_df[COL_ID].unique())
        
        # Handle Session State Selection
        default_idx = 0
        if 'map_search_tag' in st.session_state and st.session_state['map_search_tag'] in available_tags:
            default_idx = available_tags.index(st.session_state['map_search_tag'])
            
        search_tag = st.selectbox("Tag", ["None"] + available_tags, index=default_idx, label_visibility="collapsed")

    # Apply Filters
    map_df = filtered_df.copy()

    # ==========================================
    # 4. SETTINGS (Expander Layout)
    # ==========================================
    with st.expander("🛠️ Map Settings", expanded=True):
        sc1, sc2, sc3 = st.columns([1.2, 1, 1])
        
        with sc1:
            st.write("**Color Mode:**")
            color_mode = st.radio(
                "Color Mode", 
                ["Thinning Candidates (Action)", "Post-Thinning Scenario", "Mortality Risk (Heatmap)"], 
                label_visibility="collapsed"
            )
            
        with sc2:
            st.write("**Filters:**")
            min_dbh = st.slider("Hide Small Trees (< cm):", 0, 50, 0) # Default 0 to show ALL trees
            
            # [NEW] Dynamic Risk Filter
            selected_risk_levels = ["Low", "Medium", "High"] # Default to all
            if "Mortality" in color_mode:
                selected_risk_levels = st.multiselect(
                    "Show Risk Levels:",
                    ["Low", "Medium", "High"],
                    default=["Low", "Medium", "High"]
                )

        with sc3:
            st.write("**Visuals:**")
            opacity = st.slider("Canopy Opacity:", 0.1, 1.0, 0.8)
            isolate_mode = st.checkbox("🔍 Isolate Red Dots", value=False)

    # Filter by Size
    map_df = map_df[map_df[COL_CURRENT] >= min_dbh]

    # [NEW] Filter by Risk Level (only in Mortality Mode)
    if "Mortality" in color_mode:
        if 'Mortality_Risk' not in map_df.columns:
            map_df['Mortality_Risk'] = 0.0
            
        # Build Filter Condition
        risk_condition = pd.Series(False, index=map_df.index)
        
        if "High" in selected_risk_levels:
            risk_condition |= (map_df['Mortality_Risk'] > 0.5)
        if "Medium" in selected_risk_levels:
            risk_condition |= ((map_df['Mortality_Risk'] > 0.2) & (map_df['Mortality_Risk'] <= 0.5))
        if "Low" in selected_risk_levels:
            risk_condition |= (map_df['Mortality_Risk'] <= 0.2)
            
        map_df = map_df[risk_condition]

    # ==========================================
    # 5. LAYERS & LOGIC
    # ==========================================
    
    # --- COLORS ---
    def get_color(row):
        tid = str(row[COL_ID])
        is_candidate = tid in thinning_ids
        alpha = int(opacity * 255)

        # 1. Search Highlight (Cyan)
        if tid == str(search_tag): return [0, 255, 255, 255]
        
        # 2. Post-Thinning (Hide Candidates)
        if "Post-Thinning" in color_mode:
            if is_candidate: return [0, 0, 0, 0] # Invisible
            return [50, 200, 100, alpha] # Green

        # 3. Thinning Mode (Red vs Green)
        if "Thinning" in color_mode:
            if is_candidate: return [255, 0, 0, 255] # Red
            return [50, 200, 100, alpha] # Green

        # 4. Mortality Heatmap
        if "Mortality" in color_mode:
            risk = row.get('Mortality_Risk', 0)
            if risk > 0.5: return [255, 0, 0, alpha]     # Red (High)
            if risk > 0.2: return [255, 165, 0, alpha]   # Orange/Yellow (Medium)
            return [50, 200, 100, alpha]                 # Green (Low)
        
        return [50, 200, 100, alpha]

    if not map_df.empty:
        map_df['color'] = map_df.apply(get_color, axis=1)
        
        # --- SORTING (Critical for Red Dots Visibility) ---
        # Draw Green (0) -> Red (1) -> Selected (2)
        map_df['sort_priority'] = 0
        map_df.loc[map_df[COL_ID].astype(str).isin(thinning_ids), 'sort_priority'] = 1
        if str(search_tag) != "None":
            map_df.loc[map_df[COL_ID].astype(str) == str(search_tag), 'sort_priority'] = 2
            
        map_df = map_df.sort_values('sort_priority', ascending=True)

        # --- ISOLATE MODE ---
        if isolate_mode:
            # Hide everything except candidates and selection
            map_df = map_df[map_df['sort_priority'] > 0]
    
    else:
        if "Mortality" in color_mode and not selected_risk_levels:
             st.warning("⚠️ Please select at least one Risk Level to see trees.")

    layers = []

    if not map_df.empty:
        # Layer A: 3D TREES (Base Layer)
        layers.append(pdk.Layer(
            "SimpleMeshLayer",
            data=map_df,
            mesh="https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/trips/tree.obj",
            get_position=["lon", "lat"],
            get_color="color",
            get_orientation=[0, 0, 90],
            get_scale=[1, 1, 1],
            size_scale=30, 
            pickable=True,
        ))

        # Layer B: 2D DOTS (Overlay for Visibility)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=5,
            radius_min_pixels=3,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 100],
            line_width_min_pixels=1
        ))

    # ==========================================
    # 6. CAMERA & RENDER
    # ==========================================
    PASOH_ANCHOR = {"latitude": 2.9788, "longitude": 102.3131, "zoom": 16, "pitch": 45}
    
    # Auto-Zoom Logic
    # 1. Base location: Average of displayed data
    if not map_df.empty:
        init_lat = map_df['lat'].mean()
        init_lon = map_df['lon'].mean()
        init_zoom = 16
    else:
        init_lat = PASOH_ANCHOR["latitude"]
        init_lon = PASOH_ANCHOR["longitude"]
        init_zoom = PASOH_ANCHOR["zoom"]

    # 2. Override if specific tree search
    if str(search_tag) != "None":
        target = filtered_df[filtered_df[COL_ID].astype(str) == str(search_tag)]
        if not target.empty:
            init_lat = target.iloc[0]['lat']
            init_lon = target.iloc[0]['lon']
            init_zoom = 19
            st.toast(f"📍 Found Tree {search_tag}")
    
    view_state = pdk.ViewState(
        latitude=init_lat,
        longitude=init_lon,
        zoom=init_zoom,
        pitch=PASOH_ANCHOR["pitch"]
    )

    tooltip = {"html": "<b>ID:</b> {TAG}<br><b>Species:</b> {SP}<br><b>DBH:</b> {D05} cm"}
    
    if not map_df.empty:
        map_df['TAG'] = map_df[COL_ID]
        map_df['SP'] = map_df[COL_SPECIES]
        map_df['D05'] = map_df[COL_CURRENT]

    try:
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider=map_provider,
            map_style=map_style,
            api_keys={"mapbox": mapbox_key} if mapbox_key else None,
            tooltip=tooltip
        )
        st.pydeck_chart(r, use_container_width=True)
        
        if map_provider == "carto":
            st.caption("ℹ️ Using Backup Map (Mapbox Key not detected).")

    except Exception as e:
        st.error(f"Render Error: {e}")

else:
    st.info("👋 Please load your data on the Home Page first.")

# ==========================================
# NAVIGATION
# ==========================================
st.markdown("---")
col_n1, col_n2 = st.columns([1, 1])

with col_n1:
    # Go back to the Dashboard
    if st.button("⬅️ Back to Dashboard"): 
        st.switch_page("pages/0_Dashboard.py")

with col_n2:
    # Go forward to Individual Growth Analysis
    if st.button("View Predicted Growth ➡️"): 
        st.switch_page("pages/2_Individual_Growth.py")