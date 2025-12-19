# ForestManager_app.py
import streamlit as st
from src.utils import load_model_resources, load_and_process_data, run_predictions
from src.components import render_sidebar_filters, render_thinning_controls
from src.config import *

st.set_page_config(page_title="ForestManager | Home", layout="wide")

# 1. Load Backend (with Spinner)
with st.spinner('Loading Forest Data and Models...'):
    df = load_and_process_data(DATA_FILENAME)
    model, encoder = load_model_resources()

if df is not None and model is not None:
    # 2. Process Logic
    # We check if prediction is already done to save time
    if 'Predicted_Growth' not in df.columns:
        with st.spinner('Running AI Growth Predictions...'):
            df = run_predictions(df, model, encoder)
            st.session_state['df'] = df # Share with other pages

    # 3. Render Sidebar
    selected_quads, selected_species = render_sidebar_filters(df)
    growth_pct, ci_limit, prox_limit = render_thinning_controls(df)

    # 4. Apply Filtering
    df_filtered = df[
        (df['QUAD'].isin(selected_quads)) & 
        (df['SP'].isin(selected_species))
    ]
    
    # 5. Apply Thinning Rules
    if 'Predicted_Growth' in df_filtered.columns:
        growth_thresh = df_filtered['Predicted_Growth'].quantile(growth_pct / 100.0)
        
        conditions = (
            (df_filtered['Predicted_Growth'] <= growth_thresh) &
            (df_filtered['Competition_Index'] >= ci_limit) &
            (df_filtered['Nearest_Neighbor_Dist'] <= prox_limit)
        )
        df_thinning = df_filtered[conditions]
        st.session_state['df_thinning_recs'] = df_thinning

        # 6. Main Dashboard
        st.title("ForestManager Dashboard")
        
        # Metric Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Growth Threshold", f"≤ {growth_thresh:.2f} cm", help=f"Bottom {growth_pct}%")
        col2.metric("Competition Index", f"≥ {ci_limit}", help="Hegyi's Index")
        col3.metric("Proximity Threshold", f"≤ {prox_limit} m")

        if not df_thinning.empty:
            st.success(f"Found **{len(df_thinning)}** trees recommended for thinning.")
            
            # --- NEW: Download Button ---
            csv = df_thinning.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Thinning List (CSV)",
                data=csv,
                file_name='thinning_candidates.csv',
                mime='text/csv',
            )
            
            st.dataframe(df_thinning[['TAG', 'SP', 'QUAD', 'Predicted_Growth', 'Competition_Index', 'Nearest_Neighbor_Dist', 'D19']])
        else:
            st.warning("No trees match these strict criteria. Try relaxing the Competition Index or Proximity sliders.")
else:
    st.error("System Error: Could not load data or model. Run 'python train_model.py' first.")