# pages/2_Individual_Growth.py
import streamlit as st
import pandas as pd
import altair as alt
from src.config import *
import numpy as np
import joblib

st.set_page_config(page_title="ForestManager | Growth Analysis", layout="wide")

st.title("📈 Individual Tree Growth Analysis")

# Load Models specifically for simulation
@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILENAME)

model = get_model()

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # 1. SELECT TREE
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
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
        selected_tag = st.selectbox("Select Tree Tag:", filtered_tags)

    # 2. PREPARE DATA
    tree_data = df[df[COL_ID] == selected_tag].iloc[0]
    
    # --- A. Historical Data (Extended Timeline) ---
    # Loops through 1995, 2000, 2005, 2010
    chart_points = []
    last_measured_year = None
    last_measured_val = None
    
    for year, col_name in sorted(COL_YEARS.items()):
        if col_name in df.columns:
            val = tree_data[col_name]
            if pd.notna(val) and val > 0:
                chart_points.append({"Year": year, "DBH": val, "Type": "Measured"})
                last_measured_year = year
                last_measured_val = val

    # --- B. Simulation Logic ---
    st.markdown("### 🔮 Prediction Scenarios")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    with col_sim1:
        simulate_thinning = st.toggle("Simulate Thinning Effect", value=False, 
            help="Predicts how this tree would grow if we reduced competition by 50%.")
    
    # 1. Standard Prediction (Status Quo)
    predictions = []
    if 'Predicted_Size' in tree_data:
        pred_val = tree_data['Predicted_Size']
        predictions.append({"Year": 2015, "DBH": pred_val, "Type": "Predicted (Status Quo)"})
    
    # 2. Thinned Prediction (Comparison)
    if simulate_thinning and model:
        # Create a synthetic feature row with reduced Competition
        features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
        
        # Check if we have all needed features
        if all(f in tree_data for f in features):
            row_vals = tree_data[features].copy()
            
            # --- THE SIMULATION ---
            # Reduce Competition Index by 50% (Simulating neighbor removal)
            row_vals['Competition_Index'] = row_vals['Competition_Index'] * 0.5 
            
            # Predict
            pred_thin_val = model.predict(pd.DataFrame([row_vals]))[0]
            # Ensure no shrinking
            pred_thin_val = max(pred_thin_val, tree_data[COL_CURRENT])
            
            predictions.append({"Year": 2015, "DBH": pred_thin_val, "Type": "Predicted (After Thinning)"})
            
            improvement = pred_thin_val - tree_data['Predicted_Size']
            if improvement > 0:
                st.success(f"📉 Removing competition could add **+{improvement:.2f} cm** to growth!")

    # Combine Data for Plotting
    # We add the anchor point (last measured) to connect lines
    plot_data = pd.DataFrame(chart_points)
    
    pred_lines = []
    if last_measured_year:
        for p in predictions:
            # Add start point (2010) and end point (2015) for each line
            line_data = [
                {"Year": last_measured_year, "DBH": last_measured_val, "Type": p["Type"]},
                p
            ]
            pred_lines.append(pd.DataFrame(line_data))

    # --- C. Visualization ---
    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        if not plot_data.empty:
            # 1. Measured History (Solid Line)
            base = alt.Chart(plot_data).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Year:O', axis=alt.Axis(title="Year")),
                y=alt.Y('DBH', scale=alt.Scale(zero=False), axis=alt.Axis(title="Diameter (cm)")),
                color=alt.value("#2e7d32"), # Green
                tooltip=['Year', 'DBH', 'Type']
            )
            
            final_chart = base
            
            # 2. Add Prediction Lines (Dashed)
            colors = {"Predicted (Status Quo)": "#d32f2f", "Predicted (After Thinning)": "#1976d2"}
            
            for i, line_df in enumerate(pred_lines):
                line_type = line_df.iloc[1]['Type']
                c = colors.get(line_type, "grey")
                
                pred_layer = alt.Chart(line_df).mark_line(
                    point=True, strokeDash=[5, 5], strokeWidth=3
                ).encode(
                    x='Year:O', y='DBH',
                    color=alt.value(c),
                    tooltip=['Year', 'DBH', 'Type']
                )
                final_chart += pred_layer

            st.altair_chart(final_chart.properties(height=400, title=f"Growth Trajectory: Tree #{selected_tag}"), use_container_width=True)
            
            # Custom Legend
            st.caption("""
            **Legend:** <span style='color:#2e7d32'><b>———</b> Measured History</span> &nbsp;|&nbsp; 
            <span style='color:#d32f2f'><b>- - -</b> Status Quo Prediction</span> &nbsp;|&nbsp; 
            <span style='color:#1976d2'><b>- - -</b> Simulated Thinning</span>
            """, unsafe_allow_html=True)

    # --- D. Stats Panel ---
    with col_stats:
        st.subheader("Tree Statistics")
        st.info(f"Species: **{tree_data.get(COL_SPECIES, 'Unknown')}**")
        
        # Risk Meter
        if 'Mortality_Risk' in tree_data:
            risk = tree_data['Mortality_Risk'] * 100
            color = "red" if risk > 50 else ("orange" if risk > 20 else "green")
            st.metric("Mortality Risk", f"{risk:.1f}%", delta="High" if risk > 50 else "Low", delta_color="inverse")
            st.progress(int(100 - risk))
            st.caption(f"Survival Probability: {100-risk:.1f}%")

        st.divider()
        st.write("**Measurements:**")
        for p in chart_points:
            st.text(f"{p['Year']}: {p['DBH']:.2f} cm")

    # Navigation
    st.markdown("---")
    c1, c2 = st.columns([1,1])
    if c1.button("⬅️ Back to Map"): st.switch_page("pages/1_Spatial_Map.py")
    if c2.button("Back to Dashboard 🏠"): st.switch_page("pages/0_Dashboard.py")

else:
    st.warning("⚠️ Data not loaded.")