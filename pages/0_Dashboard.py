# pages/0_Dashboard.py
import streamlit as st
from src.config import * # Import constants

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("🌳 ForestManager Dashboard")

if 'df' in st.session_state:
    df = st.session_state['df']

    # ==========================================
    # 1. CONFIGURATION TABS
    # ==========================================
    st.markdown("### 🛠️ Configuration")
    tab_data, tab_rules = st.tabs(["1️⃣ Data Selection", "2️⃣ Thinning Parameters"])
    
    with tab_data:
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        
        # --- COLUMN 1: SPECIES GROUP ---
        with col_d1:
            if COL_SPECIES_GRP in df.columns:
                all_groups = sorted(df[COL_SPECIES_GRP].dropna().unique())
                # Default selects ALL groups initially
                selected_groups = st.multiselect("Select Species Group(s):", all_groups, default=all_groups)
            else:
                selected_groups = []

        # --- COLUMN 2: SPECIFIC SPECIES (Updated with Select All) ---
        with col_d2:
            if COL_SPECIES in df.columns:
                # Filter available species based on selected groups
                if COL_SPECIES_GRP in df.columns and selected_groups:
                    available_species = df[df[COL_SPECIES_GRP].isin(selected_groups)][COL_SPECIES].unique()
                else:
                    available_species = df[COL_SPECIES].unique()
                
                all_species = sorted(available_species)
                
                # --- NEW: SELECT ALL CHECKBOX ---
                use_all_species = st.checkbox("✅ Select All Species", value=False)
                
                if use_all_species:
                    # If checked: Select everything and disable the box (visual feedback)
                    selected_species = all_species
                    st.multiselect(
                        f"Select Species ({len(all_species)} available):", 
                        all_species, 
                        default=all_species, 
                        disabled=True, 
                        key="sp_select_disabled"
                    )
                else:
                    # If unchecked: Allow manual selection (Default to first 5 to save space)
                    default_sp = all_species[:5] if len(all_species) > 5 else all_species
                    selected_species = st.multiselect(
                        f"Select Species ({len(all_species)} available):", 
                        all_species, 
                        default=default_sp,
                        key="sp_select_enabled"
                    )
            else:
                selected_species = []
        
        # --- COLUMN 3: RESET ---
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with tab_rules:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            growth_pct = st.slider(
                "Remove trees growing slower than (Percentile):", 
                0, 100, 
                DEFAULT_GROWTH_PERCENTILE
            )
        with col_r2:
            max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
            ci_limit = st.slider("Minimum Competition Index (Hegyi):", 0, max_ci, 0)

    st.divider()

    # ==========================================
    # 2. LOGIC APPLICATION
    # ==========================================
    
    # Filter 1: Species Group
    if COL_SPECIES_GRP in df.columns:
        df_filtered = df[df[COL_SPECIES_GRP].isin(selected_groups)]
    else:
        df_filtered = df.copy()

    # Filter 2: Specific Species
    df_filtered = df_filtered[df_filtered[COL_SPECIES].isin(selected_species)]
    
    # Filter 3: Thinning Rules
    if 'Predicted_Growth' in df_filtered.columns and not df_filtered.empty:
        growth_thresh = df_filtered['Predicted_Growth'].quantile(growth_pct / 100.0)
        
        conditions = (
            (df_filtered['Predicted_Growth'] <= growth_thresh) &
            (df_filtered['Competition_Index'] >= ci_limit) 
        )
        
        df_thinning = df_filtered[conditions]
        st.session_state['df_thinning_recs'] = df_thinning

        # ==========================================
        # 3. RESULTS
        # ==========================================
        st.subheader("📊 Simulation Results")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Candidates for Removal", f"{len(df_thinning)} Trees")
        m_col2.metric("Growth Cutoff", f"≤ {growth_thresh:.3f}")
        m_col3.metric("Competition Floor", f"Index ≥ {ci_limit}")

        if not df_thinning.empty:
            st.success(f"Strategy identified **{len(df_thinning)}** trees.")
            
            # Download
            csv = df_thinning.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, 'thinning_candidates.csv', 'text/csv', type="primary")
            
            # Table
            display_cols = [COL_ID, COL_SPECIES_GRP, COL_SPECIES, 'Predicted_Growth', 'Competition_Index', COL_CURRENT]
            final_cols = [c for c in display_cols if c in df_thinning.columns]
            
            st.dataframe(df_thinning[final_cols], use_container_width=True, height=300)
        else:
            st.warning("No trees match the current criteria.")
            st.session_state['df_thinning_recs'] = df_filtered.head(0)
    else:
        st.warning("No data found (or no species selected).")
        st.session_state['df_thinning_recs'] = df.head(0)

    # ==========================================
    # 4. NAVIGATION
    # ==========================================
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Home"): st.switch_page("ForestManager_app.py")
    with col_n2:
        if st.button("View Map ➡️"): st.switch_page("pages/1_Spatial_Map.py")