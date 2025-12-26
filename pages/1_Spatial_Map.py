# pages/1_Spatial_Map.py
import streamlit as st
import pydeck as pdk
import pandas as pd
from src.config import * # Import your config constants (COL_CURRENT, COL_X, etc.)

st.set_page_config(page_title="ForestManager | 3D Map", layout="wide")

st.title("🗺️ 3D Forest Structure Visualization")

# Check if data is loaded in Session State
if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    
    # Handle case where thinning list might be empty
    if not st.session_state['df_thinning_recs'].empty:
        thinning_tags = set(st.session_state['df_thinning_recs'][COL_ID])
    else:
        thinning_tags = set()

    # ==========================================
    # 1. MAP CONTROLS
    # ==========================================
    with st.expander("🛠️ Map Settings & Controls", expanded=True):
        col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
        
        with col_c1:
            view_mode = st.radio(
                "Scenario View:", 
                ["Current Forest (Highlight Targets)", "Post-Thinning Scenario"], 
                horizontal=True
            )
        
        with col_c2:
            elevation_scale = st.slider(
                "Height Scale (Exaggeration):", 
                0.1, 10.0, 2.0, 0.1, 
                help="Multiplies the tree height to make differences more visible."
            )
            
        with col_c3:
            radius_size = st.slider(
                "Tree Thickness (m):", 
                0.1, 10.0, 3.0, 0.1
            )

    # ==========================================
    # 2. DATA PREPARATION
    # ==========================================
    # Ensure numerical data for coordinates and size
    df_all[COL_X] = pd.to_numeric(df_all[COL_X], errors='coerce')
    df_all[COL_Y] = pd.to_numeric(df_all[COL_Y], errors='coerce')
    df_all[COL_CURRENT] = pd.to_numeric(df_all[COL_CURRENT], errors='coerce').fillna(0)
    
    # Drop invalid rows
    df_all = df_all.dropna(subset=[COL_X, COL_Y])

    if view_mode == "Post-Thinning Scenario":
        # Show ONLY remaining trees
        plot_data = df_all[~df_all[COL_ID].isin(thinning_tags)].copy()
        # Color: Forest Green [R, G, B, A]
        plot_data['color'] = [[34, 139, 34, 200]] * len(plot_data)
        st.success(f"Visualizing **{len(plot_data)}** trees remaining after thinning.")
    else:
        # Show ALL trees
        plot_data = df_all.copy()
        
        # Color Logic: Red for Removal, Green for Keep
        def get_color(tag):
            if tag in thinning_tags:
                return [200, 30, 30, 255] # Red (Opaque)
            return [34, 139, 34, 160]     # Green (Translucent)
            
        plot_data['color'] = plot_data[COL_ID].apply(get_color)
        
        count_remove = len(df_all[df_all[COL_ID].isin(thinning_tags)])
        st.info(f"Visualizing full forest. **Red columns** indicate {count_remove} trees marked for removal.")

    # ==========================================
    # 3. RENDER PYDECK MAP
    # ==========================================
    if not plot_data.empty:
        # Calculate center point for the camera
        mid_x = plot_data[COL_X].mean()
        mid_y = plot_data[COL_Y].mean()
        
        # Define the 3D Column Layer
        layer = pdk.Layer(
            "ColumnLayer",
            data=plot_data,
            get_position=[COL_X, COL_Y],  # Uses Config constants
            get_elevation=COL_CURRENT,    # Uses 'D05' (from config)
            elevation_scale=elevation_scale,
            radius=radius_size,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        # Set the Initial Camera View
        view_state = pdk.ViewState(
            longitude=mid_x,
            latitude=mid_y,
            zoom=16,
            pitch=45,
            bearing=0
        )

        # Tooltip content on hover
        tooltip = {
            "html": f"<b>TAG:</b> {{{COL_ID}}} <br/> "
                    f"<b>Species:</b> {{{COL_SPECIES}}} <br/> "
                    f"<b>Size ({COL_CURRENT}):</b> {{{COL_CURRENT}}} cm",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        # Render the Deck
        r = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
        )
        
        st.pydeck_chart(r)
        
    else:
        st.error("Error: Dataset is empty after filtering.")

    # ==========================================
    # 4. NAVIGATION
    # ==========================================
    st.markdown("---")
    col_nav1, col_nav3 = st.columns([1, 1])
    
    with col_nav1:
        if st.button("⬅️ Back to Dashboard"):
            st.switch_page("pages/0_Dashboard.py")
            
    with col_nav3:
        if st.button("Go to Growth Trends ➡️"):
            st.switch_page("pages/2_Individual_Growth.py")
    
else:
    st.warning("⚠️ Data not loaded. Please go to the **Home** page first to initialize the system.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")