# pages/0_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.config import *

st.set_page_config(page_title="ForestManager | Dashboard", layout="wide")

st.title("🌳 ForestManager Dashboard")

# --- HELPER: PERSIST WIDGET STATE ---
def persist(key):
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
                for k in ["saved_groups", "saved_use_all", "saved_species", "saved_growth_pct", "saved_ci_level"]:
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
            # --- SIMPLIFIED COMPETITION SLIDER ---
            CI_LEVELS = ["Low", "Medium", "High"]
            CI_THRESHOLDS = {
                "Low": 2.0,   
                "Medium": 5.0, 
                "High": 8.0    
            }
            
            saved_level = st.session_state.get("saved_ci_level", "Medium")
            
            ci_level_selected = st.select_slider(
                "Minimum Competition Level:",
                options=CI_LEVELS,
                value=saved_level,
                key="ci_level",
                on_change=persist, args=("ci_level",),
                help="Select the intensity of competition stress required to flag a tree."
            )
            
            ci_limit = CI_THRESHOLDS[ci_level_selected]
            st.caption(f"Filtering trees with **Hegyi Index ≥ {ci_limit}**")

    st.divider()

    # ==========================================
    # 2. LOGIC (Wait for User Input)
    # ==========================================
    if not selected_groups and not selected_species and not use_all_species:
        st.info("👈 **Start by selecting a Species Group or Species** in the configuration panel above.")
        st.caption("The dashboard is waiting for your input to generate the analysis.")
        st.session_state['df_thinning_recs'] = df.head(0) 
    else:
        # Standard Logic
        if COL_SPECIES_GRP in df.columns and selected_groups:
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
            m_col3.metric("Competition Floor", f"{ci_level_selected} (≥ {ci_limit})")
            m_col4.metric(
                "Avg. Mortality Risk", 
                f"{avg_risk:.1f}%", 
                delta="High Risk" if avg_risk > 50 else "Normal",
                delta_color="inverse" if avg_risk > 50 else "normal"
            )

            # ==========================================
            # 🆕 MONTE CARLO RISK ASSESSMENT
            # ==========================================
            if not df_thinning.empty:
                st.markdown("---")
                st.subheader("🎲 Monte Carlo Risk Assessment")
                
                with st.expander("Run Monte Carlo Simulation (100 Iterations)", expanded=False):
                    st.write("This simulation 'rolls the dice' for every tree based on its Mortality Risk to predict the range of possible remaining stock.")
                    
                    if st.button("🚀 Run Simulation"):
                        if 'Mortality_Risk' in df_thinning.columns:
                            
                            # 1. Setup
                            n_simulations = 100
                            results = []
                            total_trees = len(df_thinning)
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            
                            # 2. Run Loop (Vectorized for Speed)
                            probs = df_thinning['Mortality_Risk'].values
                            
                            # We simulate the surviving biomass (Stock)
                            # Assuming you want to know how much stock remains if you DON'T thin them (Status Quo risk)
                            # Or if you DO thin them (Removal). 
                            # Usually MC is used to see "How many of these candidates will die anyway?"
                            
                            current_vals = df_thinning[COL_CURRENT].values
                            
                            for i in range(n_simulations):
                                # Generate random number 0.0 to 1.0 for every tree
                                random_rolls = np.random.rand(total_trees)
                                
                                # If Random Roll > Risk, Tree Survives
                                survivors = random_rolls > probs
                                
                                # Calculate surviving biomass of this specific group
                                surviving_biomass = np.sum(current_vals[survivors])
                                
                                results.append(surviving_biomass)
                                progress_bar.progress((i + 1) / n_simulations)
                                
                            # 3. Visualize Results (Histogram)
                            results = np.array(results)
                            
                            # Calculate Confidence Intervals
                            worst_case = np.percentile(results, 5) # 5th percentile
                            best_case = np.percentile(results, 95) # 95th percentile
                            avg_case = np.mean(results)
                            
                            mc_col1, mc_col2 = st.columns([1, 2])
                            
                            with mc_col1:
                                st.markdown("##### Predicted Survivor Stock")
                                st.success(f"**Best Case (95%):**\n{best_case:,.0f} cm total DBH")
                                st.info(f"**Likely Outcome:**\n{avg_case:,.0f} cm total DBH")
                                st.error(f"**Worst Case (5%):**\n{worst_case:,.0f} cm total DBH")
                                
                            with mc_col2:
                                # Simple Histogram using Altair
                                hist_data = pd.DataFrame({'Total Biomass (DBH Sum)': results})
                                
                                chart = alt.Chart(hist_data).mark_bar().encode(
                                    alt.X("Total Biomass (DBH Sum)", bin=alt.Bin(maxbins=20), title="Total Remaining Biomass (cm)"),
                                    y=alt.Y('count()', title="Frequency"),
                                    tooltip=['count()']
                                ).properties(title="Distribution of Probabilistic Outcomes")
                                
                                st.altair_chart(chart, use_container_width=True)
                        else:
                            st.error("Model must include Mortality Risk for Monte Carlo simulation.")

            # ==========================================
            # 4. EXPORT & TABLE
            # ==========================================
            if not df_thinning.empty:
                st.markdown("---")
                st.success(f"Strategy identified **{len(df_thinning)}** trees.")
                
                csv = df_thinning.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, 'thinning_candidates.csv', 'text/csv', type="primary")
                
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