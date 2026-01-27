# ForestManager_app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
from src.utils import load_and_process_data, load_model_resources, run_predictions
from src.config import DATA_FILENAME, DEFAULT_MIN_DBH
from src.utils import load_css

# 1. GLOBAL CONFIGURATION (Must be first)
st.set_page_config(page_title="ForestManager", layout="wide")

load_css()

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
        help="Trees smaller than this are removed from the analysis."
    )

# ==========================================
# 3. DATA LOADING & PREDICTION
# ==========================================
df = load_and_process_data(DATA_FILENAME, min_dbh_input)
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

    # --- ROW 1: WELCOME (Left) & MAP (Right) ---
    row1_col1, row1_col2 = st.columns([3, 2], gap="large")

    with row1_col1:
        st.markdown("### 👋 Welcome")
        tree_count = len(df) if df is not None else 0
        
        # --- EXPANDED INTRODUCTION HERE ---
        st.markdown(
            f"""
            **ForestManager** is an advanced Digital Twin designed to assist foresters in optimizing 
            ecosystem health through data-driven **silvicultural thinning**. 
            
            By integrating inventory data with machine learning, this system allows you to:
            * **Visualize Stand Structure:** Explore the forest in a 3D interactive map to identify crowding.
            * **Simulate Growth:** Predict future tree diameter and mortality risk using XGBoost algorithms.
            * **Compare Strategies:** Test different thinning intensities to maximize yield while maintaining biodiversity.
            * **Benchmark AI:** Run tournaments between different algorithms (Linear Regression vs. Random Forest) to find the best predictor.

            **Current Data Status:**
            * **Trees Loaded:** {tree_count:,}
            * **Min DBH Filter:** {min_dbh_input} cm
            """
        )
        st.info("👈 Use the sidebar to adjust global settings before starting.")

    with row1_col2:
        st.markdown("### 📍 Study Site")
        st.caption(
            "**Pasoh Forest Reserve** (Negeri Sembilan, Malaysia). "
            "A 50-hectare lowland dipterocarp research plot managed by FRIM."
        )

        # MAP RENDERING
        ICON_URL = "https://img.icons8.com/plasticine/100/000000/marker.png"
        icon_data = {"url": ICON_URL, "width": 128, "height": 128, "anchorY": 128}
        pasoh_coords = pd.DataFrame({
            'lat': [2.982], 'lon': [102.313],
            'name': ["Pasoh Forest Reserve"],
            'icon_data': [icon_data]
        })

        icon_layer = pdk.Layer(
            type="IconLayer",
            data=pasoh_coords,
            get_icon="icon_data",
            get_size=4,
            size_scale=15,
            get_position='[lon, lat]',
            pickable=True
        )

        view_state = pdk.ViewState(latitude=2.982, longitude=102.313, zoom=10, pitch=0)
        tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}

        st.pydeck_chart(
            pdk.Deck(map_style=None, initial_view_state=view_state, layers=[icon_layer], tooltip=tooltip),
            use_container_width=True
        )

    st.markdown("---")

    # --- ROW 2: HOW TO USE (Left) & METRICS (Right) ---
    row2_col1, row2_col2 = st.columns([3, 2], gap="large")

    with row2_col1:
        st.markdown("### 🚀 How to Use This App")
        st.markdown(
            """
            1.  **Configure (Dashboard)**: Select the species group(s) and species of interest.
            2.  **Simulate**: Adjust the *Growth Percentile* and *Competition Index* sliders to define your thinning strategy.
            3.  **Visualize (Spatial Map)**: Toggle between the "Current" and "Post-Thinning" views to see the physical impact.
            4.  **Analyze (AI Models)**: Compare algorithms like Random Forest vs. XGBoost to validate prediction accuracy.
            """
        )

        st.write("")
        if st.session_state.get('model_loaded'):
            if st.button("🏁 Start Analysis (Go to Dashboard)", type="primary", use_container_width=True):
                st.switch_page("pages/0_Dashboard.py")
        else:
            st.error("System Error: Data or Model could not be loaded.")

    with row2_col2:
        st.markdown("### 🧠 Understanding the Metrics")
        
        with st.expander("📉 Predicted Growth Percentile", expanded=False):
            st.write("""
            **What it is:** A ranking of how fast a tree is expected to grow compared to the rest of the forest.
            - **0-20th Percentile:** Stagnant/Suppressed trees. (Prime candidates for removal).
            - **80-100th Percentile:** High-vigor, dominant trees. (Should typically be protected).
            
            *Use this filter to quickly isolate the underperforming trees that are not contributing to forest biomass.*
            """)

        with st.expander("⚔️ Hegyi's Competition Index (CI)", expanded=False):
            st.write("""
            **What it is:** A mathematical score representing the crowding pressure on a specific tree.
            
            **The Formula:**
            $$CI_i = \sum_{j=1}^{n} \\left( \\frac{DBH_j}{DBH_i} \\times \\frac{1}{Dist_{ij}} \\right)$$
            *Where $DBH_j$ is the neighbor size, $DBH_i$ is the subject size, and $Dist$ is the distance.*
            
            **Interpretation:**
            - **CI < 3:** Low Competition (Free growing).
            - **CI 3 - 6:** Moderate Competition.
            - **CI > 8:** High Competition (Tree is likely under severe stress).
            """)

        with st.expander("🍂 Mortality Risk", expanded=False):
            st.write("""
            **What it is:** The likelihood (0% to 100%) that this specific tree will die within the next 5-year cycle.
            
            **Risk Levels:**
            - 🟢 **Low Risk (< 20%):** Tree is healthy and likely to survive.
            - 🟠 **Medium Risk (20% - 50%):** Tree is showing signs of stress.
            - 🔴 **High Risk (> 50%):** Tree is critically stressed and highly likely to die.
            
            **Key Drivers:**
            The AI calculates this based on **stopped growth** (stagnation), **high competition** (overcrowding), and **small size** (sapling vulnerability).
            """)
        with st.expander("🎲 Monte Carlo Simulation", expanded=False):
            st.write("""
            **Why we use it:** Nature is unpredictable. Even the best AI cannot predict weather or disease perfectly.
            
            **How it works:**
            Instead of giving one single guess, we run the growth prediction **100 times** with slight random variations (noise).
            - **Shaded Area:** Represents the **95% Confidence Interval** (The range where the true growth is most likely to fall).
            - **Dashed Line:** The most probable average outcome.
            """)

    st.markdown("---")
    st.caption("Developed for Final Year Project / Univesiti Putra Malaysia (UPM) © 2026")

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