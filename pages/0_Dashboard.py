# pages/0_Dashboard.py
import streamlit as st
from src.config import *

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("🌳 ForestManager Dashboard")

if 'df' in st.session_state:
    df = st.session_state['df']

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    st.markdown("### 🛠️ Configuration")
    tab_data, tab_rules = st.tabs(["1️⃣ Data Selection", "2️⃣ Thinning Parameters"])
    
    with tab_data:
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        
        # Species Group
        with col_d1:
            if COL_SPECIES_GRP in df.columns:
                all_groups = sorted(df[COL_SPECIES_GRP].dropna().unique())
                selected_groups = st.multiselect("Select Species Group(s):", all_groups, default=all_groups)
            else:
                selected_groups = []

        # Species (Select All Logic)
        with col_d2:
            if COL_SPECIES in df.columns:
                if COL_SPECIES_GRP in df.columns and selected_groups:
                    available_species = df[df[COL_SPECIES_GRP].isin(selected_groups)][COL_SPECIES].unique()
                else:
                    available_species = df[COL_SPECIES].unique()
                
                all_species = sorted(available_species)
                
                use_all_species = st.checkbox("Select All Species", value=False)
                if use_all_species:
                    selected_species = all_species
                    st.multiselect("Select Species:", all_species, default=all_species, disabled=True, key="sp_dis")
                else:
                    default_sp = all_species[:5] if len(all_species) > 5 else all_species
                    selected_species = st.multiselect("Select Species:", all_species, default=default_sp, key="sp_en")
            else:
                selected_species = []
        
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with tab_rules:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            growth_pct = st.slider("Remove trees growing slower than (Percentile):", 0, 100, DEFAULT_GROWTH_PERCENTILE)
        with col_r2:
            max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
            ci_limit = st.slider("Minimum Competition Index (Hegyi):", 0, max_ci, 0)

    st.divider()

    # ==========================================
    # 2. LOGIC
    # ==========================================
    if COL_SPECIES_GRP in df.columns:
        df_filtered = df[df[COL_SPECIES_GRP].isin(selected_groups)]
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered[df_filtered[COL_SPECIES].isin(selected_species)]
    
    if 'Predicted_Growth' in df_filtered.columns and not df_filtered.empty:
        growth_thresh = df_filtered['Predicted_Growth'].quantile(growth_pct / 100.0)
        
        conditions = (
            (df_filtered['Predicted_Growth'] <= growth_thresh) &
            (df_filtered['Competition_Index'] >= ci_limit) 
        )
        
        df_thinning = df_filtered[conditions].copy()
        
        # --- CALCULATE MORTALITY FOR DISPLAY ---
        if 'Mortality_Risk' in df_thinning.columns:
            # We round percentage to 1 decimal place as it's standard for %
            df_thinning['Mortality Risk (%)'] = (df_thinning['Mortality_Risk'] * 100).round(1)
            avg_risk = df_thinning['Mortality Risk (%)'].mean()
        else:
            df_thinning['Mortality Risk (%)'] = 0.0
            avg_risk = 0.0

        st.session_state['df_thinning_recs'] = df_thinning

        # ==========================================
        # 3. RESULTS
        # ==========================================
        st.subheader("📊 Simulation Results")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric("Candidates for Removal", f"{len(df_thinning)} Trees")
        m_col2.metric("Growth Cutoff", f"≤ {growth_thresh:.3f}")
        m_col3.metric("Competition Floor", f"Index ≥ {ci_limit}")
        
        m_col4.metric(
            "Avg. Mortality Risk", 
            f"{avg_risk:.1f}%", 
            delta="High Risk" if avg_risk > 50 else "Normal",
            delta_color="inverse" if avg_risk > 50 else "normal"
        )

        if not df_thinning.empty:
            st.success(f"Strategy identified **{len(df_thinning)}** trees.")
            
            csv = df_thinning.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, 'thinning_candidates.csv', 'text/csv', type="primary")
            
            # --- TABLE DISPLAY PREPARATION ---
            display_cols = [
                COL_ID, 
                COL_SPECIES_GRP, 
                COL_SPECIES, 
                'Predicted_Growth', 
                'Competition_Index', 
                'Mortality Risk (%)', 
                COL_CURRENT
            ]
            final_cols = [c for c in display_cols if c in df_thinning.columns]
            
            # Create a clean display copy
            df_display = df_thinning[final_cols].copy()
            
            # --- ROUNDING LOGIC (Max 4 Decimals) ---
            # We specifically target the float columns that tend to have long decimals
            cols_to_round = ['Predicted_Growth', 'Competition_Index']
            for c in cols_to_round:
                if c in df_display.columns:
                    df_display[c] = df_display[c].round(4)
            
            st.dataframe(
                df_display, 
                use_container_width=True, 
                height=300
            )
        else:
            st.warning("No trees match the current criteria.")
            st.session_state['df_thinning_recs'] = df_filtered.head(0)
    else:
        st.warning("No data found.")
        st.session_state['df_thinning_recs'] = df.head(0)

    # Navigation
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Home"): st.switch_page("ForestManager_app.py")
    with col_n2:
        if st.button("View Map ➡️"): st.switch_page("pages/1_Spatial_Map.py")