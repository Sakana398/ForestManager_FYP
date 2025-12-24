# ForestManager_app.py
import streamlit as st
from src.utils import load_and_process_data, load_model_resources, run_predictions
from src.config import DATA_FILENAME

st.set_page_config(page_title="ForestManager | Home", layout="wide")

# ==========================================
# 1. GLOBAL DATA LOADING (System Init)
# ==========================================
with st.spinner('Initializing ForestManager System...'):
    if 'df' not in st.session_state:
        df = load_and_process_data(DATA_FILENAME)
        model, encoder = load_model_resources()
        
        if df is not None and model is not None:
            if 'Predicted_Growth' not in df.columns:
                df = run_predictions(df, model, encoder)
            
            st.session_state['df'] = df
            st.session_state['model_loaded'] = True
        else:
            st.session_state['model_loaded'] = False

# ==========================================
# 2. HERO SECTION
# ==========================================
st.title("🌲 ForestManager")
st.subheader("Intelligent Silvicultural Decision Support System")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 👋 Welcome")
    st.markdown(
        """
        **ForestManager** is an AI-powered tool designed to assist foresters in optimizing 
        forest health through data-driven **silvicultural thinning**. 
        
        By analyzing historical growth patterns, spatial density, and competition indices, 
        this application helps identify which trees are struggling and where thinning 
        could benefit the overall stand.
        """
    )
    
    st.markdown("### 🚀 How to Use This App")
    st.markdown(
        """
        1.  **Configure (Dashboard)**: Select your forest quadrants and species of interest.
        2.  **Simulate**: Adjust the *Growth Percentile* and *Competition Index* sliders to define your thinning strategy.
        3.  **Visualize (Spatial Map)**: Toggle between the "Current" and "Post-Thinning" views to see the physical impact on the forest structure.
        4.  **Analyze (Individual Growth)**: Drill down into specific trees to view their historical performance and predicted future growth.
        """
    )

    st.info("💡 **Tip:** Start by filtering for the bottom 25% of growth and high competition to find the most stressed trees.")

    # --- NAVIGATION: Home -> Dashboard ---
    st.write("")
    if st.session_state.get('model_loaded'):
        if st.button("🏁 Start Analysis (Go to Dashboard)", type="primary", use_container_width=True):
            st.switch_page("pages/0_Dashboard.py")
    else:
        st.error("System Error: Data or Model could not be loaded. Please check your files.")

with col2:
    st.markdown("### 🧠 Understanding the Metrics")
    
    with st.expander("📉 Predicted Growth Percentile", expanded=True):
        st.write(
            """
            The AI model predicts the diameter growth of every tree for the next cycle. 
            Filtering by **Percentile** allows you to target the slowest growers.
            * *Example:* Selecting **20%** targets the bottom 20% of trees with the lowest predicted growth.
            """
        )

    with st.expander("⚔️ Hegyi's Competition Index (CI)", expanded=True):
        st.write(
            """
            This index measures the stress a tree is under from its neighbors.
            $$ CI_i = \sum (D_j / D_i) / DIST_{ij} $$
            * **High CI:** The tree is small and surrounded by large, close neighbors (High Stress).
            * **Low CI:** The tree is dominant or isolated (Low Stress).
            """
        )

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Forest_thinning_in_Finland.JPG/640px-Forest_thinning_in_Finland.JPG", 
        caption="Forest Thinning Operation (Illustrative)",
        use_container_width=True
    )

st.markdown("---")
st.caption("ForestManager FYP v2.0 | Powered by Random Forest Regression & Streamlit")