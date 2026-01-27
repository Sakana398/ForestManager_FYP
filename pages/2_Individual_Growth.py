# pages/2_Individual_Growth.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
from src.config import *

st.set_page_config(page_title="ForestManager | Growth Analysis", layout="wide")

st.title("📈 Individual Tree Growth Analysis")

# Load Models
@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILENAME)

model = get_model()

# --- HELPER: MONTE CARLO PREDICTION ---
def predict_with_uncertainty(model, input_row, baseline_size, volatility=0.2, iterations=100):
    """
    Predicts future size by adding modeled growth to a BASELINE size.
    baseline_size: The starting size (e.g., D10) to add growth to.
    """
    # Ensure input is a proper DataFrame
    input_df = pd.DataFrame([input_row])
    
    raw_increment = model.predict(input_df)[0]
    
    # Enforce Non-Negative Growth (Trees don't shrink)
    base_increment = max(raw_increment, 0.0)
    
    np.random.seed(42)
    simulations = np.random.normal(loc=base_increment, scale=abs(base_increment * volatility) + 0.01, size=iterations)
    simulations = np.maximum(simulations, 0)
    
    inc_lower = np.percentile(simulations, 5)
    inc_upper = np.percentile(simulations, 95)
    
    return {
        "mean": baseline_size + base_increment, # Add to the correct baseline (2010)
        "min": baseline_size + inc_lower,
        "max": baseline_size + inc_upper,
        "increment": base_increment
    }

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Check Strategy State
    if 'df_thinning_recs' in st.session_state and not st.session_state['df_thinning_recs'].empty:
        thinning_ids = set(st.session_state['df_thinning_recs'][COL_ID].astype(str))
        strategy_active = True
        strategy_count = len(thinning_ids)
    else:
        thinning_ids = set()
        strategy_active = False
        strategy_count = 0

    # ==========================================
    # 1. SELECT TREE & SIZE CONTEXT
    # ==========================================
    col_sel1, col_sel2 = st.columns([1, 2])
    
    with col_sel1:
        st.subheader("Select Target")
        if strategy_active:
            if st.button("✨ Find Tree with Removable Neighbors"):
                candidates = df[
                    (~df[COL_ID].astype(str).isin(thinning_ids)) &
                    (df['Competition_Index'] > 4)
                ] 
                if not candidates.empty:
                    best_tree = candidates.sample(1).iloc[0][COL_ID]
                    st.session_state['selected_tree_id'] = best_tree
                    st.rerun()
                else:
                    st.warning("No suitable candidate found.")

        all_tags = sorted(df[COL_ID].unique())
        if COL_SPECIES in df.columns:
            all_species = sorted(df[COL_SPECIES].unique())
            sel_species = st.selectbox("Filter Species:", ["All"] + all_species)
            if sel_species != "All":
                filtered_tags = sorted(df[df[COL_SPECIES] == sel_species][COL_ID].unique())
            else:
                filtered_tags = all_tags
        else:
            filtered_tags = all_tags
            
        idx = 0
        if 'selected_tree_id' in st.session_state and st.session_state['selected_tree_id'] in filtered_tags:
            idx = filtered_tags.index(st.session_state['selected_tree_id'])
            
        selected_tag = st.selectbox("Select Tree Tag:", filtered_tags, index=idx)
        st.session_state['selected_tree_id'] = selected_tag

    # GET DATA
    tree_data = df[df[COL_ID] == selected_tag].iloc[0]

    with col_sel2:
        if COL_CURRENT in df.columns:
            st.subheader("Population Distribution")
            
            df['DBH_Class'] = (df[COL_CURRENT] // 5) * 5
            dist_data = df['DBH_Class'].value_counts().reset_index()
            dist_data.columns = ['DBH_Class', 'Count']
            
            current_tree_class = (tree_data[COL_CURRENT] // 5) * 5
            
            base_hist = alt.Chart(dist_data).mark_bar(color='#e0e0e0').encode(
                x=alt.X('DBH_Class:O', title='DBH Class (cm)', sort='ascending'),
                y=alt.Y('Count', title='Tree Count'),
                tooltip=['DBH_Class', 'Count']
            )
            
            highlight_data = dist_data[dist_data['DBH_Class'] == current_tree_class]
            highlight_hist = alt.Chart(highlight_data).mark_bar(color='#d32f2f').encode(
                x='DBH_Class:O', y='Count', tooltip=['DBH_Class', 'Count']
            )
            
            final_hist = (base_hist + highlight_hist).properties(height=180)
            st.altair_chart(final_hist, use_container_width=True)
            
            percentile = (df[COL_CURRENT] < tree_data[COL_CURRENT]).mean() * 100
            st.caption(f"📍 This tree ({tree_data[COL_CURRENT]:.1f} cm) is larger than **{percentile:.1f}%** of the forest.")

    st.divider()

    # ==========================================
    # 2. HISTORICAL DATA
    # ==========================================
    chart_points = []
    last_measured_year = None
    last_measured_val = None
    
    # We loop through history to find the most recent valid size (D10)
    for year, col_name in sorted(COL_YEARS.items()):
        if col_name in df.columns:
            val = tree_data[col_name]
            if pd.notna(val) and val > 0:
                chart_points.append({"Year": year, "DBH": val, "Type": "Measured"})
                last_measured_year = year
                last_measured_val = val # This captures the 2010 size

    # ==========================================
    # 3. THINNING SIMULATION
    # ==========================================
    st.markdown("### 🔮 Prediction Scenarios") 
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        simulate_thinning = st.toggle(
            "Apply Dashboard Thinning Strategy",
            value=False,
            disabled=not strategy_active,
            help="Recalculates growth by physically removing neighbors marked in the Dashboard."
        )
        
        if not strategy_active:
            st.caption("⚠️ No thinning strategy selected on Dashboard.")
        else:
            st.caption(f"✅ Strategy Loaded: {strategy_count} trees marked for removal.")

    features = [
        COL_CURRENT, 'GROWTH_HIST', 
        'Nearest_Neighbor_Dist', 'Local_Density', 
        'Competition_Index', 'Interaction_Vigor_Comp', 
        'SP_Encoded'
    ]
    
    predictions = []
    
    # Defensively fill missing columns
    for col in features:
        if col not in tree_data:
            tree_data[col] = 0.0

    if all(f in tree_data for f in features) and last_measured_val is not None:
        # Use D05 features for the RATE, but add to D10 for the PROJECTION
        current_dbh_for_rate = tree_data[COL_CURRENT] 
        input_row = tree_data[features].copy()
        
        # [CRITICAL] Use last_measured_val (D10) as the baseline for 2015
        baseline_for_projection = last_measured_val 
        
        # A. PREDICT STATUS QUO
        res_sq = predict_with_uncertainty(model, input_row, baseline_for_projection)
        predictions.append({
            "Year": 2015,
            "DBH": res_sq['mean'],
            "Min_DBH": res_sq['min'],
            "Max_DBH": res_sq['max'],
            "Type": "Predicted (Status Quo)"
        })
        
        # B. PREDICT THINNING
        new_ci = tree_data['Competition_Index']
        removed_neighbors = 0
        total_neighbors = 0
        pred_thin_val = res_sq['mean']
        
        if simulate_thinning and strategy_active:
            c_lat = tree_data[COL_Y]
            c_lon = tree_data[COL_X]
            radius_deg = COMPETITION_RADIUS * (1/111111.0)
            
            candidates_df = df[
                (df[COL_X].between(c_lon - radius_deg*1.5, c_lon + radius_deg*1.5)) &
                (df[COL_Y].between(c_lat - radius_deg*1.5, c_lat + radius_deg*1.5)) &
                (df[COL_ID] != selected_tag)
            ].copy()
            
            simulated_ci = 0
            
            if not candidates_df.empty:
                for _, neighbor in candidates_df.iterrows():
                    d_deg = np.sqrt((neighbor[COL_X] - c_lon)**2 + (neighbor[COL_Y] - c_lat)**2)
                    
                    if d_deg <= radius_deg:
                        d_meter = d_deg * 111111.0
                        if d_meter < 0.1: d_meter = 0.1 
                        total_neighbors += 1
                        
                        if str(neighbor[COL_ID]) in thinning_ids:
                            removed_neighbors += 1
                        else:
                            n_dbh = neighbor[COL_CURRENT]
                            simulated_ci += (n_dbh / current_dbh_for_rate) / d_meter
                                
            # Update Features
            old_ci = tree_data['Competition_Index']
            new_ci = simulated_ci
            input_row['Competition_Index'] = new_ci

            if old_ci > 0:
                species_factor = tree_data['Interaction_Vigor_Comp'] / old_ci
                input_row['Interaction_Vigor_Comp'] = new_ci * species_factor
            else:
                input_row['Interaction_Vigor_Comp'] = 0.0

            # Run Prediction (Adding to the 2010 baseline)
            res_thin = predict_with_uncertainty(model, input_row, baseline_for_projection)
            pred_thin_val = res_thin['mean']
            
            predictions.append({
                "Year": 2015,
                "DBH": res_thin['mean'],
                "Min_DBH": res_thin['min'],
                "Max_DBH": res_thin['max'],
                "Type": "Predicted (After Thinning)"
            })
            
            with col_sim2:
                st.markdown("#### Simulation Results")
                gain = pred_thin_val - res_sq['mean']
                
                if removed_neighbors > 0:
                    st.success(f"**Simulation Active:** Removed {removed_neighbors} of {total_neighbors} neighbors.")
                else:
                    st.warning(f"**No Change:** Found {total_neighbors} neighbors, but NONE are in the removal list.")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Old CI", f"{tree_data['Competition_Index']:.2f}")
                col_res2.metric("New CI", f"{new_ci:.2f}", delta=f"-{tree_data['Competition_Index']-new_ci:.2f}", delta_color="inverse")
                col_res3.metric("Growth Gain", f"+{gain:.2f} cm")
            

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    plot_data = pd.DataFrame(chart_points)
    pred_lines = []
    
    if last_measured_year:
        for p in predictions:
            line_data = [
                {
                    "Year": last_measured_year,
                    "DBH": last_measured_val,
                    "Min_DBH": last_measured_val,
                    "Max_DBH": last_measured_val,
                    "Type": p["Type"]
                },
                p
            ]
            pred_lines.append(pd.DataFrame(line_data))

    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        st.subheader("Growth Forecast with Uncertainty")
        if not plot_data.empty:
            base = alt.Chart(plot_data).mark_line(point=True, strokeWidth=3).encode( 
                x=alt.X('Year:O', axis=alt.Axis(title="Year")),
                y=alt.Y('DBH', scale=alt.Scale(zero=False), axis=alt.Axis(title="Diameter (cm)")),
                color=alt.value("#2e7d32"), 
                tooltip=['Year', 'DBH', 'Type']
            )
            final_chart = base
            
            colors = {"Predicted (Status Quo)": "#d32f2f", "Predicted (After Thinning)": "#1976d2"}
            
            for line_df in pred_lines:
                line_type = line_df.iloc[1]['Type']
                c = colors.get(line_type, "grey")
                
                line_layer = alt.Chart(line_df).mark_line(
                    point=True, strokeDash=[5, 5], strokeWidth=3
                ).encode(
                    x='Year:O', y='DBH', color=alt.value(c), tooltip=['Year', 'DBH', 'Type']
                )
                
                band_layer = alt.Chart(line_df).mark_area(opacity=0.2).encode(
                    x='Year:O', y='Min_DBH', y2='Max_DBH', color=alt.value(c)
                )
                
                final_chart += band_layer + line_layer

            st.altair_chart(final_chart.properties(height=450), use_container_width=True)
            
            st.caption("""
            **Legend:** <span style='color:#2e7d32'><b>———</b> Measured History</span> &nbsp;|&nbsp; 
            <span style='color:#d32f2f'><b>- - -</b> Status Quo Prediction</span> &nbsp;|&nbsp; 
            <span style='color:#1976d2'><b>- - -</b> Simulated Thinning</span>
            <br><i>Shaded areas represent the 95% Confidence Interval (Monte Carlo Simulation).</i>
            """, unsafe_allow_html=True)

    with col_stats:
        st.subheader("Tree Statistics")
        st.write(f"**Species:** {tree_data.get(COL_SPECIES, 'Unknown')}")
        if 'Mortality_Risk' in tree_data:
            risk = tree_data['Mortality_Risk'] * 100
            
            # [FIXED] Force "- Low" to trigger Green Down Arrow
            if risk > 50:
                risk_label = "+ High"
            else:
                risk_label = "- Low"
            
            st.metric("Mortality Risk", f"{risk:.1f}%", delta=risk_label, delta_color="inverse")
            st.progress(int(100 - risk))
        
        st.divider()
        st.write(f"**Competition Index:** {tree_data['Competition_Index']:.2f}")

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    if c1.button("⬅️ Back to Map"): st.switch_page("pages/1_Spatial_Map.py")
    if c2.button("Back to Dashboard 🏠"): st.switch_page("pages/0_Dashboard.py")

else:
    st.warning("⚠️ Data not loaded.")