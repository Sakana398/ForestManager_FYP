# ForestManager_app.py
import streamlit as st
from src.utils import load_and_process_data, load_model_resources, run_predictions
from src.config import DATA_FILENAME

st.set_page_config(page_title="ForestManager | Home", layout="wide")

# ==========================================
# 1. GLOBAL DATA LOADING
# ==========================================
# We load data here so it is available for all pages (Dashboard, Map, Growth)
with st.spinner('Initializing ForestManager System...'):
    # Load raw data
    if 'df' not in st.session_state:
        df = load_and_process_data(DATA_FILENAME)
        model, encoder = load_model_resources()
        
        # Pre-calculate predictions if valid
        if df is not None and model is not None:
             # Check if we need to run predictions
            if 'Predicted_Growth' not in df.columns:
                df = run_predictions(df, model, encoder)
            
            st.session_state['df'] = df
            st.session_state['model_loaded'] = True
        else:
            st.session_state['model_loaded'] = False

# ==========================================
# 2. LANDING PAGE CONTENT
# ==========================================
st.title("Welcome to ForestManager")
st.subheader("AI-Driven Silvicultural Decision Support System")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🪓 What is Silvicultural Thinning?")
    st.info(
        """
        **Silvicultural thinning** is the selective removal of trees to improve the 
        growth rate and health of the remaining forest. 
        
        Crowded trees compete for resources like sunlight and water. By removing specific 
        trees (often those with poor growth potential or high competition), we allow the 
        remaining "crop trees" to reach maturity faster.
        """
    )
    
    st.markdown("### 📖 How to Use This App")
    st.markdown(
        """
        1.  **Dashboard**: Set thinning parameters (Growth Rate, Competition, Proximity).
        2.  **Spatial Map**: Visualize the forest before and after the proposed thinning.
        3.  **Growth Trends**: Deep dive into the history of individual trees.
        """
    )
    
    st.write("---")
    
    # Navigation Button
    if st.session_state.get('model_loaded'):
        if st.button("🚀 Launch Dashboard", type="primary"):
            st.switch_page("pages/0_Dashboard.py")
    else:
        st.error("System Error: Data or Model could not be loaded. Please check your files.")

with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Forest_thinning_in_Finland.JPG/640px-Forest_thinning_in_Finland.JPG", 
             caption="Forest Thinning Operation (Source: Wikimedia Commons)")
    st.warning(
        """
        **Disclaimer:** This tool uses a Random Forest model to predict future growth. 
        Recommendations should be validated by forestry experts before execution.
        """
    )