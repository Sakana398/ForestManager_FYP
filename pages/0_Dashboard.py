# pages/0_Dashboard.py
import streamlit as st
from src.config import * # Import config to get default values (DEFAULT_GROWTH_PERCENTILE, etc.)

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("ForestManager Dashboard")

# Check if data is loaded
if 'df' in st.session_state:
    df = st.session_state['df']

    # ==========================================
    # 1. FILTERS & CONTROLS (ON PAGE)
    # ==========================================
    with st.expander("**Filter & Simulation Settings**", expanded=True):
        
        # Top Row: Data Filters (Quadrants & Species)
        st.subheader("1. Data Scope")
        col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
        
        with col_f1:
            # Quadrant Selector
            if 'QUAD' in df.columns:
                all_quads = sorted(df['QUAD'].dropna().unique().astype(int))
                selected_quads = st.multiselect("Select Quadrant(s):", all_quads, default=all_quads)
            else:
                selected_quads = []

        with col_f2:
            # Species Selector
            if 'SP' in df.columns:
                all_species = sorted(df['SP'].dropna().unique())
                selected_species = st.multiselect(
                    f"Select Species ({len(all_species)} found):", 
                    all_species, 
                    default=all_species[:5] # Default to first 5
                )
            else:
                selected_species = []
        
        with col_f3:
            st.write("### Actions")
            if st.button("Reset All Filters"):
                st.session_state.clear()
                st.rerun()

        st.markdown("---")

        # Bottom Row: Thinning Simulation Controls
        st.subheader("2. Thinning Rules")
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            growth_pct = st.slider(
                "Low Growth Rate (% Below)", 
                0, 100, 
                st.session_state.get('last_growth_pct', DEFAULT_GROWTH_PERCENTILE),
                help="Select trees growing slower than this percentile."
            )

        with col_c2:
            max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
            ci_limit = st.slider(
                "High Competition (CI ≥)", 
                0, max_ci, 
                0,
                help="Hegyi's Index. Higher value = More pressure from neighbors."
            )

        with col_c3:
            prox_limit = st.slider(
                "Close Proximity (Distance ≤)", 
                0.0, 10.0, 
                DEFAULT_PROXIMITY,
                help="Target trees physically crowding others."
            )

    # ==========================================
    # 2. APPLY LOGIC
    # ==========================================
    
    # Filter Data
    df_filtered = df[
        (df['QUAD'].isin(selected_quads)) & 
        (df['SP'].isin(selected_species))
    ]
    
    # Apply Thinning Rules
    if 'Predicted_Growth' in df_filtered.columns:
        # Dynamic Threshold Calculation
        growth_thresh = df_filtered['Predicted_Growth'].quantile(growth_pct / 100.0)
        
        conditions = (
            (df_filtered['Predicted_Growth'] <= growth_thresh) &
            (df_filtered['Competition_Index'] >= ci_limit) &
            (df_filtered['Nearest_Neighbor_Dist'] <= prox_limit)
        )
        
        df_thinning = df_filtered[conditions]
        
        # Save results to session state
        st.session_state['df_thinning_recs'] = df_thinning

        # ==========================================
        # 3. METRICS & RESULTS
        # ==========================================
        st.write("### Simulation Results")
        
        # Metric Row
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Growth Threshold", f"≤ {growth_thresh:.2f} cm", help=f"Bottom {growth_pct}% of trees")
        m_col2.metric("Competition Index", f"≥ {ci_limit}", help="Hegyi's Index")
        m_col3.metric("Proximity Threshold", f"≤ {prox_limit} m")

        if not df_thinning.empty:
            st.success(f"**Recommendation:** Remove {len(df_thinning)} trees based on current criteria.")
            
            # CSV Download
            csv = df_thinning.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Thinning List (CSV)",
                data=csv,
                file_name='thinning_candidates.csv',
                mime='text/csv',
            )
            
            # Data Preview
            st.dataframe(
                df_thinning[['TAG', 'SP', 'QUAD', 'Predicted_Growth', 'Competition_Index', 'Nearest_Neighbor_Dist', 'D19']],
                use_container_width=True
            )
        else:
            st.warning("No trees match these strict criteria. Try relaxing the sliders above.")
            # Clear state to avoid stale data on map
            st.session_state['df_thinning_recs'] = df_filtered.head(0)
else:
    st.warning("Data not loaded. Please go to the **Home** page first to initialize the system.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")