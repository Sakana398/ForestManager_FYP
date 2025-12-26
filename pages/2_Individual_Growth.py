# pages/2_Individual_Growth.py
import streamlit as st
import pandas as pd
import altair as alt
from src.config import *

st.set_page_config(page_title="ForestManager | Growth Analysis", layout="wide")

st.title("📈 Individual Tree Growth Analysis")

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # ==========================================
    # 1. SELECT TREE
    # ==========================================
    col_sel1, col_sel2 = st.columns([1, 2])
    
    with col_sel1:
        # Search box for Tree ID
        all_tags = sorted(df[COL_ID].unique())
        
        # Optional: Species Filter
        if COL_SPECIES in df.columns:
            all_species = sorted(df[COL_SPECIES].unique())
            sel_species = st.selectbox("Filter by Species (Optional):", ["All"] + all_species)
            
            if sel_species != "All":
                filtered_tags = sorted(df[df[COL_SPECIES] == sel_species][COL_ID].unique())
            else:
                filtered_tags = all_tags
        else:
            filtered_tags = all_tags
            
        selected_tag = st.selectbox("Select Tree Tag:", filtered_tags)

    # ==========================================
    # 2. PREPARE DATA FOR CHART
    # ==========================================
    tree_data = df[df[COL_ID] == selected_tag].iloc[0]
    
    # Define timeline mapping (Adjust if your config columns map to different years)
    # History (2000), Current (2005), Target (2010)
    timeline_map = {
        2000: COL_HISTORY, 
        2005: COL_CURRENT, 
        2010: COL_TARGET
    }
    
    # 1. Build Historical Data (Green Line)
    history_points = []
    last_measured_year = None
    last_measured_val = None
    
    for year, col_name in timeline_map.items():
        if col_name in df.columns:
            val = tree_data[col_name]
            if pd.notna(val) and val > 0:
                history_points.append({"Year": year, "DBH": val, "Type": "Measured"})
                # Track the last valid point to connect the red line later
                last_measured_year = year
                last_measured_val = val

    df_history = pd.DataFrame(history_points)

    # 2. Build Predicted Data (Red Line)
    # This line must START at the last measured point and END at the prediction
    prediction_points = []
    
    if 'Predicted_Size' in tree_data and pd.notna(tree_data['Predicted_Size']):
        pred_val = tree_data['Predicted_Size']
        pred_year = 2015 # Assuming 5-year step
        
        if last_measured_year is not None:
            # Anchor point (Start of Red Line)
            prediction_points.append({"Year": last_measured_year, "DBH": last_measured_val, "Type": "Predicted"})
            # Future point (End of Red Line)
            prediction_points.append({"Year": pred_year, "DBH": pred_val, "Type": "Predicted"})
            
    df_pred = pd.DataFrame(prediction_points)

    # ==========================================
    # 3. VISUALIZATION (Multi-Layer Chart)
    # ==========================================
    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        if not df_history.empty:
            # Layer A: Historical Line (Green)
            chart_hist = alt.Chart(df_history).mark_line(
                point=True, 
                color='#2e7d32', # Green
                strokeWidth=3
            ).encode(
                x=alt.X('Year:O', axis=alt.Axis(title="Year")),
                y=alt.Y('DBH', scale=alt.Scale(zero=False), axis=alt.Axis(title="Diameter (cm)")),
                tooltip=['Year', 'DBH', 'Type']
            )
            
            # Layer B: Predicted Line (Red)
            # We only add this layer if we have prediction data
            if not df_pred.empty:
                chart_pred = alt.Chart(df_pred).mark_line(
                    point=True, 
                    color='#d32f2f', # Red
                    strokeDash=[5, 5], # Dashed line for effect
                    strokeWidth=3
                ).encode(
                    x=alt.X('Year:O'),
                    y=alt.Y('DBH'),
                    tooltip=['Year', 'DBH', 'Type']
                )
                
                # Combine layers
                final_chart = (chart_hist + chart_pred).properties(
                    title=f"Growth Progression: Tree #{selected_tag}",
                    height=400
                )
            else:
                final_chart = chart_hist.properties(title=f"Growth History: Tree #{selected_tag}")

            st.altair_chart(final_chart, use_container_width=True)
            
            # Legend / Key
            st.caption("🟢 **Solid Green Line:** Historical Measurement | 🔴 **Dashed Red Line:** AI Prediction")
            
        else:
            st.warning("No historical data available for this tree.")

    with col_stats:
        st.subheader("Tree Details")
        st.write(f"**Species:** {tree_data.get(COL_SPECIES, 'Unknown')}")
        if COL_SPECIES_GRP in df.columns:
            st.write(f"**Group:** {tree_data.get(COL_SPECIES_GRP, '-')}")
            
        st.divider()
        
        # Display Growth Rate
        if 'Predicted_Growth' in tree_data:
            pg = tree_data['Predicted_Growth']
            st.metric(
                "Predicted Growth (5yr)", 
                f"{pg:.2f} cm",
                delta="Future Trend"
            )
            
        if 'Competition_Index' in tree_data:
            st.metric("Competition Index", f"{tree_data['Competition_Index']:.2f}")

        # Show Values Table
        st.markdown("#### Data Points")
        # Combine for display
        display_rows = history_points + [p for p in prediction_points if p['Year'] == 2015]
        for row in display_rows:
            label = "🔮 Predicted" if row['Year'] == 2015 else "📏 Measured"
            st.text(f"{row['Year']}: {row['DBH']:.2f} cm  ({label})")

    # ==========================================
    # 4. NAVIGATION
    # ==========================================
    st.markdown("---")
    col_nav1, col_nav3 = st.columns([1, 1])
    with col_nav1:
        if st.button("⬅️ Back to Map"):
            st.switch_page("pages/1_Spatial_Map.py")
    with col_nav3:
        if st.button("Back to Dashboard 🏠"):
            st.switch_page("pages/0_Dashboard.py")

else:
    st.warning("⚠️ Data not loaded. Please go to the **Home** page first.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")