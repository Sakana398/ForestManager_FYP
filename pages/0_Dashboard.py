# pages/0_Dashboard.py
import streamlit as st
from src.config import *

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("🌳 ForestManager Dashboard")

# --- HELPER: PERSIST WIDGET STATE ---
def persist(key):
    """Save widget state to session_state immediately on change."""
    if key in st.session_state:
        st.session_state[f"saved_{key}"] = st.session_state[key]

if 'df' in st.session_state:
    df = st.session_state['df']

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    st.markdown("### 🛠️ Configuration")
    tab_data, tab_rules = st.tabs(["1️⃣ Data Selection", "2️⃣ Thinning Parameters"])
    
    with tab_data:
        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
        
        # --- SPECIES GROUP ---
        with col_d1:
            if COL_SPECIES_GRP in df.columns:
                all_groups = sorted(df[COL_SPECIES_GRP].dropna().unique())
                
                # CHANGED: Default is now empty list [] instead of all_groups
                default_groups = st.session_state.get("saved_groups", [])
                default_groups = [g for g in default_groups if g in all_groups]
                
                selected_groups = st.multiselect(
                    "Select Species Group(s):", 
                    all_groups, 
                    default=default_groups,
                    key="groups", 
                    on_change=persist, args=("groups",)
                )
            else:
                selected_groups = []

        # --- SPECIES ---
        with col_d2:
            if COL_SPECIES in df.columns:
                if COL_SPECIES_GRP in df.columns and selected_groups:
                    available_species = df[df[COL_SPECIES_GRP].isin(selected_groups)][COL_SPECIES].unique()
                else:
                    available_species = df[COL_SPECIES].unique()
                
                all_species = sorted(available_species)
                
                # Checkbox Persistence
                default_use_all = st.session_state.get("saved_use_all", False)
                use_all_species = st.checkbox(
                    "✅ Select All Species", 
                    value=default_use_all,
                    key="use_all",
                    on_change=persist, args=("use_all",)
                )
                
                if use_all_species:
                    selected_species = all_species
                    st.multiselect("Select Species:", all_species, default=all_species, disabled=True, key="sp_dis")
                else:
                    # CHANGED: Default is now empty list []
                    default_sp = st.session_state.get("saved_species", [])
                    default_sp = [s for s in default_sp if s in all_species]
                    
                    selected_species = st.multiselect(
                        "Select Species:", 
                        all_species, 
                        default=default_sp, 
                        key="species",
                        on_change=persist, args=("species",)
                    )
            else:
                selected_species = []
        
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("🔄 Reset Filters", use_container_width=True):
                for k in ["saved_groups", "saved_use_all", "saved_species", "saved_growth_pct", "saved_ci_limit"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    with tab_rules:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            def_growth = st.session_state.get("saved_growth_pct", DEFAULT_GROWTH_PERCENTILE)
            growth_pct = st.slider(
                "Remove trees growing slower than (Percentile):", 
                0, 100, def_growth,
                key="growth_pct",
                on_change=persist, args=("growth_pct",)
            )
            
        with col_r2:
            max_ci = int(df['Competition_Index'].max()) if 'Competition_Index' in df.columns else 10
            def_ci = st.session_state.get("saved_ci_limit", 0)
            def_ci = min(def_ci, max_ci)
            
            ci_limit = st.slider(
                "Minimum Competition Index (Hegyi):", 
                0, max_ci, def_ci,
                key="ci_limit",
                on_change=persist, args=("ci_limit",)
            )

    st.divider()

    # ==========================================
    # 2. LOGIC (Apply Filters)
    # ==========================================
    # Logic: If nothing is selected, we return empty so the dashboard starts blank
    if not selected_groups and not selected_species and not use_all_species:
        st.info("👈 Please select **Species Groups** or **Species** to begin analysis.")
        st.session_state['df_thinning_recs'] = df.head(0) # Empty
    else:
        if COL_SPECIES_GRP in df.columns:
            # If no group selected but species ARE selected (via all/manual), we don't filter by group
            # But if group is selected, we filter by it.
            if selected_groups:
                df_filtered = df[df[COL_SPECIES_GRP].isin(selected_groups)]
            else:
                df_filtered = df.copy() # Allow searching by species across all groups if group is empty
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
                
                # --- TABLE DISPLAY ---
                display_cols = [COL_ID, COL_SPECIES_GRP, COL_SPECIES, 'Predicted_Growth', 'Competition_Index', 'Mortality Risk (%)', COL_CURRENT]
                final_cols = [c for c in display_cols if c in df_thinning.columns]
                
                df_disp = df_thinning[final_cols].copy()
                for c in ['Predicted_Growth', 'Competition_Index']:
                    if c in df_disp.columns: df_disp[c] = df_disp[c].round(4)

                st.dataframe(df_disp, use_container_width=True, height=300)
            else:
                st.warning("No trees match the current criteria.")
                st.session_state['df_thinning_recs'] = df_filtered.head(0)
        else:
            st.warning("No data found for the selected species.")
            st.session_state['df_thinning_recs'] = df.head(0)

    # Navigation
    st.markdown("---")
    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("⬅️ Back to Home"): st.switch_page("ForestManager_app.py")
    with col_n2:
        if st.button("View Map ➡️"): st.switch_page("pages/1_Spatial_Map.py")