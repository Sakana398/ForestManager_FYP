# pages/0_Dashboard.py
import streamlit as st
from src.config import *
from src.components import render_sidebar_filters # Kept for utility, though we use custom layout here

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("🌳 ForestManager Dashboard")

# Check if data is loaded
if 'df' in st.session_state:
    df = st.session_state['df']

    # ==========================================
    # 1. CLEAN INPUT SECTION (TABS)
    # ==========================================
    st.markdown("### 🛠️ Configuration")
    
    # We use tabs to separate the "Data Slicing" from the "Thinning Logic"
    tab_data, tab_rules = st.tabs(["1️⃣ Data Selection", "2️⃣ Thinning Parameters"])
    
    # --- TAB 1: DATA SELECTION ---
    with tab_data:
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        
        with col_d1:
            if 'QUAD' in df.columns:
                all_quads = sorted(df['QUAD'].dropna().unique().astype(int))
                selected_quads = st.multiselect("Select Quadrant(s):", all_quads, default=all_quads)
            else:
                selected_quads = []

        with col_d2:
            if 'SP' in df.columns:
                all_species = sorted(df['SP'].dropna().unique())
                # Default to a few common species if too many exist, to keep UI clean
                default_sp = all_species[:5] if len(all_species) > 5 else all_species
                selected_species = st.multiselect(
                    f"Select Species ({len(all_species)} available):", 
                    all_species, 
                    default=default_sp
                )
            else:
                selected_species = []
        
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    # --- TAB 2: THINNING RULES ---
    with tab_rules:
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### 📉 Growth Target")
            growth_pct = st.slider(
                "Remove trees growing slower than (Percentile):", 
                0, 100, 
                st.session_state.get('last_growth_pct', DEFAULT_GROWTH_PERCENTILE),
                help="Setting this to 25 means the bottom 25% of trees by growth rate will be candidates."
            )
            
        with col_r2:
            st.markdown("#### ⚔️ Competition Stress")
            max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
            ci_limit = st.slider(
                "Minimum Competition Index (Hegyi):", 
                0, max_ci, 
                0,
                help="Target trees under high biological stress. Higher value = More crowded."
            )

    st.divider()

    # ==========================================
    # 2. LOGIC APPLICATION
    # ==========================================
    
    # Filter Data
    df_filtered = df[
        (df['QUAD'].isin(selected_quads)) & 
        (df['SP'].isin(selected_species))
    ]
    
    if 'Predicted_Growth' in df_filtered.columns:
        # Dynamic Threshold Calculation
        growth_thresh = df_filtered['Predicted_Growth'].quantile(growth_pct / 100.0)
        
        # Apply Conditions
        conditions = (
            (df_filtered['Predicted_Growth'] <= growth_thresh) &
            (df_filtered['Competition_Index'] >= ci_limit) 
        )
        
        df_thinning = df_filtered[conditions]
        
        # Save results to session state for the Map page
        st.session_state['df_thinning_recs'] = df_thinning

        # ==========================================
        # 3. RESULTS & METRICS
        # ==========================================
        st.subheader("📊 Simulation Results")
        
        # Key Metrics Row
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(
            label="🎯 Candidates for Removal", 
            value=f"{len(df_thinning)} Trees",
            delta=f"{len(df_thinning)/len(df_filtered)*100:.1f}% of selected" if len(df_filtered) > 0 else "0%"
        )
        m_col2.metric(
            label="Growth Cutoff (cm/yr)", 
            value=f"≤ {growth_thresh:.3f}",
            help="Trees growing slower than this are targeted."
        )
        m_col3.metric(
            label="Competition Floor", 
            value=f"Index ≥ {ci_limit}"
        )

        if not df_thinning.empty:
            st.success(f"Strategy identified **{len(df_thinning)}** trees for silvicultural thinning.")
            
            # Action Row
            ac_col1, ac_col2 = st.columns([1, 4])
            with ac_col1:
                csv = df_thinning.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download List",
                    data=csv,
                    file_name='thinning_candidates.csv',
                    mime='text/csv',
                    type="primary"
                )
            
            # Data Preview
            st.dataframe(
                df_thinning[['TAG', 'SP', 'QUAD', 'Predicted_Growth', 'Competition_Index', 'D19']],
                use_container_width=True,
                height=300
            )
        else:
            st.warning("No trees match the current criteria. Try increasing the Growth Percentile or decreasing the Competition Index.")
            st.session_state['df_thinning_recs'] = df_filtered.head(0)

    # ==========================================
    # 4. NAVIGATION
    # ==========================================
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    
    with col_nav1:
        if st.button("⬅️ Back to Home"):
            st.switch_page("ForestManager_app.py")
            
    with col_nav3:
        if st.button("View Spatial Map ➡️", type="secondary"):
            st.switch_page("pages/1_Spatial_Map.py")

else:
    st.error("⚠️ Data not loaded. Please return to the Home page to initialize the system.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")