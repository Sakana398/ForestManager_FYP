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
def predict_with_uncertainty(model, input_row, current_dbh, volatility=0.2, iterations=100):
    """
    Predicts growth increment and generates a 95% confidence interval.
    """
    # 1. Base Prediction (The Mean)
    base_increment = model.predict(pd.DataFrame([input_row]))[0]
    
    # 2. Monte Carlo Simulation (Simulate 100 possible futures)
    # We assume prediction error follows a normal distribution around the mean
    np.random.seed(42) # For consistent "randomness"
    simulations = np.random.normal(loc=base_increment, scale=abs(base_increment * volatility), size=iterations)
    
    # Clip negative growth (trees rarely shrink significantly)
    simulations = np.maximum(simulations, 0)
    
    # 3. Calculate Bounds
    inc_lower = np.percentile(simulations, 5)  # Worst Case
    inc_upper = np.percentile(simulations, 95) # Best Case
    
    return {
        "mean": current_dbh + base_increment,
        "min": current_dbh + inc_lower,
        "max": current_dbh + inc_upper,
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
    # 1. SELECT TREE
    # ==========================================
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        # AUTO-FINDER BUTTON
        if strategy_active: # Only show if strategy is active
            if st.button("✨ Find Tree with Removable Neighbors"): # Auto-select a tree with high competition and not yet thinned
                candidates = df[ # Filter candidates
                    (~df[COL_ID].astype(str).isin(thinning_ids)) &  # Not already marked for thinning
                    (df['Competition_Index'] > 4) # High competition
                ] 
                if not candidates.empty: # Select one at random
                    best_tree = candidates.sample(1).iloc[0][COL_ID] # Get the tree ID
                    st.session_state['selected_tree_id'] = best_tree # Save to session state
                    st.rerun() # Refresh to reflect change
                else: # No candidates found
                    st.warning("No suitable candidate found.") # Notify user

        all_tags = sorted(df[COL_ID].unique()) # All Tree Tags
        if COL_SPECIES in df.columns: # Species Filter
            all_species = sorted(df[COL_SPECIES].unique()) # All Species
            sel_species = st.selectbox("Filter Species:", ["All"] + all_species) # Species Dropdown
            if sel_species != "All": # Filter Tags by Species
                filtered_tags = sorted(df[df[COL_SPECIES] == sel_species][COL_ID].unique()) # Filtered Tags
            else: #
                filtered_tags = all_tags # No Filter
        else: # No Species Column
            filtered_tags = all_tags # All Tags
            
        idx = 0 # Default Index
        if 'selected_tree_id' in st.session_state and st.session_state['selected_tree_id'] in filtered_tags: # Pre-select if exists
            idx = filtered_tags.index(st.session_state['selected_tree_id']) # Get Index
            
        selected_tag = st.selectbox("Select Tree Tag:", filtered_tags, index=idx) # Tree Dropdown
        st.session_state['selected_tree_id'] = selected_tag # Save Selection

    # Get Data
    tree_data = df[df[COL_ID] == selected_tag].iloc[0] # Selected Tree Data

    # ==========================================
    # 2. HISTORICAL DATA
    # ==========================================
    chart_points = [] # For Visualization
    last_measured_year = None # To connect predictions
    last_measured_val = None # To connect predictions
    
    for year, col_name in sorted(COL_YEARS.items()): # Historical Measurements
        if col_name in df.columns: # Check Column Exists
            val = tree_data[col_name] # Get Value
            if pd.notna(val) and val > 0: # Valid Measurement
                chart_points.append({"Year": year, "DBH": val, "Type": "Measured"}) # Add to Chart
                last_measured_year = year # Update Last Year
                last_measured_val = val # Update Last Value

    # ==========================================
    # 3. THINNING SIMULATION
    # ==========================================
    st.markdown("### 🔮 Prediction Scenarios") 
    
    col_sim1, col_sim2 = st.columns([1, 2]) # Simulation Options
    
    with col_sim1: # Simulation Controls
        simulate_thinning = st.toggle( # Toggle Simulation
            "Apply Dashboard Thinning Strategy", # Toggle Label
            value=False,  # Default Off
            disabled=not strategy_active, # Disable if no strategy
            help="Recalculates growth by physically removing neighbors marked in the Dashboard." # Toggle Help
        )
        
        if not strategy_active: # No Strategy Warning
            st.caption("⚠️ No thinning strategy selected on Dashboard.") # Warning Caption
        else: # Strategy Active Caption
            st.caption(f"✅ Strategy Loaded: {strategy_count} trees marked for removal.") # Active Caption

    # PREPARE INPUTS
    features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded'] # Model Features
    predictions = [] # Store Predictions
    
    if all(f in tree_data for f in features): # Ensure all features are present
        current_dbh = tree_data[COL_CURRENT] # Current DBH
        input_row = tree_data[features].copy() # Model Input Row
        
        # A. PREDICT STATUS QUO (Red Line)
        # -------------------------------
        res_sq = predict_with_uncertainty(model, input_row, current_dbh) # Status Quo Prediction
        
        predictions.append({
            "Year": 2015, # Future Year
            "DBH": res_sq['mean'], # Predicted Mean
            "Min_DBH": res_sq['min'], # Lower Bound
            "Max_DBH": res_sq['max'], # Upper Bound
            "Type": "Predicted (Status Quo)" # Prediction Type
        })
        
        # B. PREDICT THINNING (Blue Line)
        # -------------------------------
        new_ci = tree_data['Competition_Index'] # Default to current CI
        removed_neighbors = 0  # Count of removed neighbors
        total_neighbors = 0  # Total neighbors considered
        pred_thin_val = res_sq['mean'] # Default to SQ if no thinning happens
        
        if simulate_thinning and strategy_active: # If simulation is active
            # Spatial Search
            c_lat = tree_data[COL_Y] # Current Tree Latitude
            c_lon = tree_data[COL_X] # Current Tree Longitude
            radius_deg = COMPETITION_RADIUS * (1/111111.0) # Convert to degrees
            
            candidates_df = df[     # Find Neighboring Trees
                (df[COL_X].between(c_lon - radius_deg*1.5, c_lon + radius_deg*1.5)) &  # Expand Search Box
                (df[COL_Y].between(c_lat - radius_deg*1.5, c_lat + radius_deg*1.5)) &  # Narrow Down by Y
                (df[COL_ID] != selected_tag) # Exclude Self
            ].copy() # Copy to avoid SettingWithCopyWarning
            
            simulated_ci = 0 # Recalculate CI
            
            if not candidates_df.empty:
                for _, neighbor in candidates_df.iterrows():
                    # Now neighbor is a Series, so we can use string keys like [COL_X]
                    d_deg = np.sqrt((neighbor[COL_X] - c_lon)**2 + (neighbor[COL_Y] - c_lat)**2)
                    
                    if d_deg <= radius_deg:
                        d_meter = d_deg * 111111.0
                        if d_meter < 0.1: d_meter = 0.1 
                        total_neighbors += 1
                        
                        # Ensure we convert ID to string for comparison
                        if str(neighbor[COL_ID]) in thinning_ids:
                            removed_neighbors += 1
                        else:
                            n_dbh = neighbor[COL_CURRENT]
                            simulated_ci += (n_dbh / current_dbh) / d_meter
                                
            new_ci = simulated_ci   # Update to new CI
            
            # Predict with NEW CI
            input_row['Competition_Index'] = new_ci # Update CI in Input
            res_thin = predict_with_uncertainty(model, input_row, current_dbh) # Predict Again
            pred_thin_val = res_thin['mean']   # Update Predicted Value
            
            predictions.append({    # Add to Predictions
                "Year": 2015,  # Future Year
                "DBH": res_thin['mean'],    # Predicted Mean
                "Min_DBH": res_thin['min'],    # Lower Bound
                "Max_DBH": res_thin['max'],    # Upper Bound
                "Type": "Predicted (After Thinning)" # Prediction Type
            })
            
            # Stats Display
            with col_sim2: # Results Column
                st.markdown("#### Simulation Results")
                gain = pred_thin_val - res_sq['mean']   # Growth Gain
                
                if removed_neighbors > 0: # Some neighbors were removed
                    st.success(f"**Simulation Active:** Removed {removed_neighbors} of {total_neighbors} neighbors.") # Success Message
                else:   # No neighbors removed
                    st.warning(f"**No Change:** Found {total_neighbors} neighbors, but NONE are in the removal list.") # Warning Message
                
                col_res1, col_res2, col_res3 = st.columns(3)    # Result Metrics
                col_res1.metric("Old CI", f"{tree_data['Competition_Index']:.2f}")  # Current CI
                col_res2.metric("New CI", f"{new_ci:.2f}", delta=f"-{tree_data['Competition_Index']-new_ci:.2f}", delta_color="inverse") # New CI
                col_res3.metric("Growth Gain", f"+{gain:.2f} cm") # Growth Gain
                
                # Ecological Impact (Biomass)
                old_biomass = 0.06 * (res_sq['mean'] ** 2.6) # Old Biomass Estimate
                new_biomass = 0.06 * (pred_thin_val ** 2.6)     # New Biomass Estimate
                biomass_gain = new_biomass - old_biomass # Biomass Gain
                carbon_gain = biomass_gain * 0.47   # Carbon Sequestered (47% of Biomass)

                st.markdown("---")
                st.markdown("##### 🌍 Ecological Impact") 
                c_col1, c_col2 = st.columns(2) # Ecological Metrics
                c_col1.metric("Biomass Gain", f"+{biomass_gain:.2f} kg") # Biomass Gain
                c_col2.metric("Carbon Seq.", f"+{carbon_gain:.2f} kg", help="Estimated additional carbon captured.") # Carbon Gain

    # ==========================================
    # 4. VISUALIZATION (Cone of Uncertainty)
    # ==========================================
    plot_data = pd.DataFrame(chart_points) # Historical DataFrame
    pred_lines = []    # Prediction Line Segments
    
    if last_measured_year: # If we have historical data
        for p in predictions: # Each Prediction
            # We construct a dataframe for each line segment
            line_data = [
                {
                    "Year": last_measured_year, # Connect from last measured
                    "DBH": last_measured_val, # Connect from last measured
                    "Min_DBH": last_measured_val, # No uncertainty at start
                    "Max_DBH": last_measured_val, # No uncertainty at start
                    "Type": p["Type"] # Prediction Type
                },
                p
            ]
            pred_lines.append(pd.DataFrame(line_data)) # Add to Lines

    col_viz, col_stats = st.columns([2, 1]) # Visualization & Stats
    
    with col_viz: # Visualization Column
        st.subheader("Growth Forecast with Uncertainty") # Subheader
        if not plot_data.empty: # If we have data to plot
            # 1. Base Layer (History)
            base = alt.Chart(plot_data).mark_line(point=True, strokeWidth=3).encode( 
                x=alt.X('Year:O', axis=alt.Axis(title="Year")),
                y=alt.Y('DBH', scale=alt.Scale(zero=False), axis=alt.Axis(title="Diameter (cm)")),
                color=alt.value("#2e7d32"), 
                tooltip=['Year', 'DBH', 'Type']
            )
            final_chart = base
            
            # 2. Add Cones & Dashed Lines
            colors = {"Predicted (Status Quo)": "#d32f2f", "Predicted (After Thinning)": "#1976d2"}
            
            for line_df in pred_lines:
                line_type = line_df.iloc[1]['Type']
                c = colors.get(line_type, "grey")
                
                # A. The Line
                line_layer = alt.Chart(line_df).mark_line(
                    point=True, strokeDash=[5, 5], strokeWidth=3
                ).encode(
                    x='Year:O', y='DBH', color=alt.value(c), tooltip=['Year', 'DBH', 'Type']
                )
                
                # B. The Cone (Area)
                band_layer = alt.Chart(line_df).mark_area(opacity=0.2).encode(
                    x='Year:O',
                    y='Min_DBH',
                    y2='Max_DBH',
                    color=alt.value(c)
                )
                
                final_chart += band_layer + line_layer

            st.altair_chart(final_chart.properties(height=400, title=f"Growth Forecast w/ Uncertainty (95% CI)"), use_container_width=True)
            
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
            st.metric("Mortality Risk", f"{risk:.1f}%", delta="High" if risk > 50 else "Low", delta_color="inverse")
            st.progress(int(100 - risk))
        st.divider()
        st.write(f"**Current CI:** {tree_data['Competition_Index']:.2f}")

    # ==========================================
        # DBH CLASS DISTRIBUTION CHART
        # ==========================================
        st.markdown("##### 📏 Size Comparison")
        
        if COL_CURRENT in df.columns:
            # 1. Create Bins (DBH Classes)
            # We bin trees into groups of 5cm (e.g., 10-15, 15-20)
            df['DBH_Class'] = (df[COL_CURRENT] // 5) * 5
            
            # 2. Prepare Data for Altair
            # We want a histogram of counts
            dist_data = df['DBH_Class'].value_counts().reset_index()
            dist_data.columns = ['DBH_Class', 'Count']
            
            # 3. Identify WHERE the selected tree falls
            current_tree_class = (tree_data[COL_CURRENT] // 5) * 5
            
            # 4. Create Chart
            # Base Layer: Grey Bars for population
            base_hist = alt.Chart(dist_data).mark_bar(color='#e0e0e0').encode(
                x=alt.X('DBH_Class:O', title='DBH Class (cm)', sort='ascending'),
                y=alt.Y('Count', title='Number of Trees'),
                tooltip=['DBH_Class', 'Count']
            )
            
            # Highlight Layer: Red Bar for THIS tree's class
            highlight_data = dist_data[dist_data['DBH_Class'] == current_tree_class]
            
            highlight_hist = alt.Chart(highlight_data).mark_bar(color='#d32f2f').encode(
                x='DBH_Class:O',
                y='Count',
                tooltip=['DBH_Class', 'Count']
            )
            
            # Combine
            final_hist = (base_hist + highlight_hist).properties(
                height=200, 
                title="Your Tree vs. Population"
            )
            
            st.altair_chart(final_hist, use_container_width=True)
            
            # Percentile Rank Text
            # "You are larger than 85% of trees"
            percentile = (df[COL_CURRENT] < tree_data[COL_CURRENT]).mean() * 100
            st.caption(f"This tree is larger than **{percentile:.1f}%** of the forest.")

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    if c1.button("⬅️ Back to Map"): st.switch_page("pages/1_Spatial_Map.py")
    if c2.button("Back to Dashboard 🏠"): st.switch_page("pages/0_Dashboard.py")

else:
    st.warning("⚠️ Data not loaded.")