# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from src.config import *

st.set_page_config(page_title="ForestManager | Digital Twin", layout="wide")

st.title("🗺️ Forest Digital Twin")

# --- 1. SECURE TOKEN ---
try:
    MAPBOX_KEY = st.secrets["mapbox"]["token"]
except:
    st.error("🚨 Mapbox token missing. Check .streamlit/secrets.toml")
    st.stop()

# --- 2. PASOH ANCHOR ---
PASOH_LAT = 2.9788
PASOH_LON = 102.3131

if 'df' in st.session_state:
    raw_df = st.session_state['df'].copy()
    
    # 🌍 COORDINATE FIX
    # If data is > 180, it's Meters -> Convert to GPS. 
    # If data is < 180, it's already GPS -> Use directly.
    meters_per_deg = 111139.0
    if raw_df[COL_X].max() > 180:
        raw_df['lon'] = PASOH_LON + (raw_df[COL_X] / meters_per_deg)
        raw_df['lat'] = PASOH_LAT + (raw_df[COL_Y] / meters_per_deg)
    else:
        raw_df['lon'] = raw_df[COL_X]
        raw_df['lat'] = raw_df[COL_Y]

    # Load Thinning Data
    thinning_ids = set()
    if 'df_thinning_recs' in st.session_state and not st.session_state['df_thinning_recs'].empty:
         thinning_ids = set(st.session_state['df_thinning_recs'][COL_ID].astype(str))

    # ==========================================
    # 3. FILTERS (Top Row)
    # ==========================================
    c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
    
    with c1:
        st.write("#### 🎯 Quick Find")
        if thinning_ids:
            if st.button(f"✨ Find Candidate ({len(thinning_ids)} trees)", type="primary", use_container_width=True):
                # Filter for candidates that actually exist in the map
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
        
        # Handle Selection
        default_idx = 0
        if 'map_search_tag' in st.session_state and st.session_state['map_search_tag'] in available_tags:
            default_idx = available_tags.index(st.session_state['map_search_tag'])
            
        search_tag = st.selectbox("Tag", ["None"] + available_tags, index=default_idx if 'map_search_tag' in st.session_state else 0, label_visibility="collapsed")

    # Apply Filters
    map_df = filtered_df.copy()

    # ==========================================
    # 4. MAP SETTINGS (Exact Layout Request)
    # ==========================================
    with st.expander("🛠️ Map Settings", expanded=True):
        sc1, sc2, sc3 = st.columns([1.2, 1, 1])
        
        with sc1:
            st.write("**Color Mode:**")
            color_mode = st.radio(
                "Color Mode", 
                ["Thinning Candidates (Action)", "Mortality Risk (Heatmap)", "Post-Thinning Scenario"], 
                label_visibility="collapsed"
            )
            
        with sc2:
            st.write("**Filters:**")
            min_dbh = st.slider("Hide Small Trees (< cm):", 0, 50, 0) # Default 0 to show ALL trees

        with sc3:
            st.write("**Visuals:**")
            opacity = st.slider("Canopy Opacity:", 0.1, 1.0, 0.8)
            isolate_mode = st.checkbox("🔍 Isolate Red Dots (Hide Green)", value=False)

    # Filter by Size
    map_df = map_df[map_df[COL_CURRENT] >= min_dbh]

    # ==========================================
    # 5. LOGIC & LAYERS
    # ==========================================
    
    # --- COLOR GENERATOR ---
    def get_color(row):
        tid = str(row[COL_ID])
        is_candidate = tid in thinning_ids
        alpha = int(opacity * 255)

        # 1. SEARCH HIGHLIGHT (Cyan - Always Top Priority)
        if tid == str(search_tag):
            return [0, 255, 255, 255] 

        # 2. POST-THINNING MODE (Hide Candidates)
        if "Post-Thinning" in color_mode:
            if is_candidate:
                return [0, 0, 0, 0] # Invisible
            else:
                return [50, 200, 100, alpha] # Healthy Green

        # 3. THINNING CANDIDATES MODE (Red vs Green)
        if "Thinning" in color_mode:
            if is_candidate:
                return [255, 0, 0, 255] # Red (Full Opacity)
            else:
                return [50, 200, 100, alpha] # Green

        # 4. MORTALITY RISK MODE (Heatmap)
        if "Mortality" in color_mode:
            risk = row.get('Mortality_Risk', 0)
            if risk > 0.5: return [255, 0, 0, alpha]     # High Risk (Red)
            if risk > 0.2: return [255, 165, 0, alpha]   # Med Risk (Orange)
            return [50, 200, 100, alpha]                 # Low Risk (Green)

        return [50, 200, 100, alpha]

    map_df['color'] = map_df.apply(get_color, axis=1)
    
    # --- SORTING FIX ---
    # We MUST draw green trees first, then red trees on top.
    # 0 = Green, 1 = Red, 2 = Selected
    map_df['sort_priority'] = 0
    map_df.loc[map_df[COL_ID].astype(str).isin(thinning_ids), 'sort_priority'] = 1
    if str(search_tag) != "None":
        map_df.loc[map_df[COL_ID].astype(str) == str(search_tag), 'sort_priority'] = 2
        
    map_df = map_df.sort_values('sort_priority', ascending=True)

    # --- ISOLATE MODE ---
    if isolate_mode:
        # Hide everything that isn't a candidate or selected
        map_df = map_df[map_df['sort_priority'] > 0]

    layers = []

    # LAYER A: 3D TREES (Base Layer)
    # Renders the forest structure
    layer_trees = pdk.Layer(
        "SimpleMeshLayer",
        data=map_df,
        mesh="https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/trips/tree.obj",
        get_position=["lon", "lat"],
        get_color="color",
        get_orientation=[0, 0, 90],
        get_scale=[1, 1, 1],
        size_scale=30, 
        pickable=True,
    )
    layers.append(layer_trees)

    # LAYER B: 2D DOTS (Overlay Layer)
    # Renders distinct dots ON TOP so you can see the 75 candidates clearly
    layer_dots = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=5, # Fixed size for visibility
        radius_min_pixels=3,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 100],
        line_width_min_pixels=1
    )
    layers.append(layer_dots)

    # LAYER C: SEARCH HIGHLIGHT (Pulsing Ring)
    if search_tag != "None":
        highlight_df = map_df[map_df[COL_ID].astype(str) == str(search_tag)]
        if not highlight_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=highlight_df,
                get_position=["lon", "lat"],
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[0, 255, 255, 255],
                get_radius=15,
                radius_min_pixels=10,
                stroked=True,
                line_width_min_pixels=3
            ))

    # ==========================================
    # 6. RENDER MAP
    # ==========================================
    if search_tag != "None" and not filtered_df[filtered_df[COL_ID].astype(str) == str(search_tag)].empty:
        target = filtered_df[filtered_df[COL_ID].astype(str) == str(search_tag)].iloc[0]
        view_state = pdk.ViewState(latitude=target['lat'], longitude=target['lon'], zoom=19, pitch=45)
        st.toast(f"📍 Found Tree {search_tag}")
    else:
        view_state = pdk.ViewState(latitude=PASOH_LAT, longitude=PASOH_LON, zoom=16, pitch=45)

    tooltip = {"html": "<b>ID:</b> {TAG}<br><b>Species:</b> {SP}<br><b>DBH:</b> {D05} cm"}

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style="mapbox://styles/mapbox/satellite-v9",
        api_keys={"mapbox": MAPBOX_KEY},
        tooltip=tooltip
    )
    
    st.pydeck_chart(r, use_container_width=True)

else:
    st.info("👋 Please load your data on the Home Page first.")