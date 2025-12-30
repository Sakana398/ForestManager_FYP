# ForestManager_app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
from src.utils import load_and_process_data, load_model_resources, run_predictions
from src.config import DATA_FILENAME, DEFAULT_MIN_DBH

# 1. GLOBAL CONFIGURATION (Must be first)
st.set_page_config(page_title="ForestManager", layout="wide")

# ==========================================
# 2. SIDEBAR (Global Controls)
# ==========================================
with st.sidebar:
    st.header("⚙️ Global Settings")
    st.info("These settings control the data loaded for the entire application.")
    
    # Global Filter: Minimum Tree Size
    min_dbh_input = st.slider(
        "Min. Tree Size (DBH cm):", 
        min_value=1.0, 
        max_value=20.0, 
        value=DEFAULT_MIN_DBH,
        step=0.5,
        key="global_min_dbh",
        help="Trees smaller than this are removed from the analysis to focus on established stock and prevent skewing competition metrics."
    )

# ==========================================
# 3. DATA LOADING & PREDICTION
# ==========================================
# Load Data (Cached based on min_dbh_input)
df = load_and_process_data(DATA_FILENAME, min_dbh_input)

# Load Models
model_grow, model_mort, encoder = load_model_resources()

# Run AI Predictions if needed
if df is not None and model_grow is not None:
    if 'Predicted_Growth' not in df.columns:
        with st.spinner("Running AI Analysis (Growth & Mortality)..."):
            df = run_predictions(df, model_grow, model_mort, encoder)
    
    st.session_state['df'] = df
    st.session_state['model_loaded'] = True
else:
    st.session_state['model_loaded'] = False

# ==========================================
# 4. LANDING PAGE CONTENT
# ==========================================
def landing_page():
    st.title("🌲 ForestOps")
    st.subheader("Forest Thinning Decision Support System")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 👋 Welcome")
        tree_count = len(df) if df is not None else 0
        
        st.markdown(
            f"""
            **ForestManager** is an AI-powered tool designed to assist foresters in optimizing 
            forest health through data-driven **silvicultural thinning**. 
            
            **Current Data Status:**
            * **Trees Loaded:** {tree_count:,}
            * **Min DBH Filter:** {min_dbh_input} cm
            """
        )
        
        st.markdown("### 🚀 How to Use This App")
        st.markdown(
            """
            1.  **Configure (Dashboard)**: Select the species group(s) and species of interest.
            2.  **Simulate**: Adjust the *Growth Percentile* and *Competition Index* sliders to define your thinning strategy.
            3.  **Visualize (Spatial Map)**: Toggle between the "Current" and "Post-Thinning" views to see the physical impact on the forest structure.
            4.  **Analyze (Individual Growth)**: Drill down into specific trees to view their historical performance, **Mortality Risk**, and predicted future growth.
            """
        )

        st.write("")
        if st.session_state.get('model_loaded'):
            if st.button("🏁 Start Analysis (Go to Dashboard)", type="primary", use_container_width=True):
                st.switch_page("pages/0_Dashboard.py")
        else:
            st.error("System Error: Data or Model could not be loaded. Please check your files.")

    with col2:
        st.markdown("### 🧠 Understanding the Metrics")
        
        with st.expander("📉 Predicted Growth Percentile", expanded=False):
            st.write(
                """
                The AI model predicts the diameter growth of every tree for the next cycle. 
                Filtering by **Percentile** allows you to target the slowest growers.
                * *Example:* Selecting **20%** targets the bottom 20% of trees with the lowest predicted growth.
                """
            )

        with st.expander("⚔️ Hegyi's Competition Index (CI)", expanded=False):
            st.write(
                """
                This index measures the stress a tree is under from its neighbors.
                $$ CI_i = \sum (D_j / D_i) / DIST_{ij} $$
                * **High CI:** The tree is small and surrounded by large, close neighbors (High Stress).
                * **Low CI:** The tree is dominant or isolated (Low Stress).
                """
            )
            
        with st.expander("💀 Mortality Risk", expanded=False):
            st.write(
                """
                The probability (0-100%) that a tree will die in the next cycle based on its current competition and slow growth.
                * **High Risk (>50%):** Immediate attention needed.
                """
            )

    # --- LOCATION MAP SECTION ---
    st.markdown("---")
    st.subheader("📍 Study Site Location")
    
    col_map_desc, col_map_view = st.columns([1, 3])
    
    with col_map_desc:
        st.info(
            """
            **Pasoh Forest Reserve**
            
            *Negeri Sembilan, Malaysia*
            
            A 50-hectare research plot managed by FRIM. This lowland dipterocarp forest serves as the primary dataset for this model.
            """
        )
    
    with col_map_view:
        # 1. Define Data (Single Point)
        ICON_URL = "https://img.icons8.com/plasticine/100/000000/marker.png"

        icon_data = {
            "url": ICON_URL,
            "width": 128,
            "height": 128,
            "anchorY": 128
        }

        pasoh_coords = pd.DataFrame({
            'lat': [2.982],
            'lon': [102.313],
            'name': ["Pasoh Forest Reserve"],
            'icon_data': [icon_data]
        })

        # 2. Define Layer
        icon_layer = pdk.Layer(
            type="IconLayer",
            data=pasoh_coords,
            get_icon="icon_data",
            get_size=4,
            size_scale=15,
            get_position='[lon, lat]',
            pickable=True
        )

        # 3. View State
        view_state = pdk.ViewState(
            latitude=2.982,
            longitude=102.313,
            zoom=11,
            pitch=0
        )
        
        # 4. Tooltip
        tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}

        # 5. Render (FIXED: map_style=None uses Streamlit default, preventing errors)
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,  # <--- CHANGED FROM 'mapbox://...' TO None
                initial_view_state=view_state,
                layers=[icon_layer],
                tooltip=tooltip
            )
        )

    st.markdown("---")
    st.caption("ForestManager FYP v2.0 | Powered by XGBoost & Streamlit")

# ==========================================
# 5. NAVIGATION SETUP
# ==========================================
pages = [
    st.Page(landing_page, title="Home", icon="🏠"),
    st.Page("pages/0_Dashboard.py", title="Analysis Dashboard", icon="📊"),
    st.Page("pages/1_Spatial_Map.py", title="3D Forest Map", icon="🗺️"),
    st.Page("pages/2_Individual_Growth.py", title="Growth Trends", icon="📈"),
]

pg = st.navigation(pages)
pg.run()