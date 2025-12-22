# pages/2_Individual_Growth.py
import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(page_title="ForestManager | Growth Trends", layout="wide")

st.title("📈 Individual Tree Growth Trends")

# Check for FULL data
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Selection
    tree_tag_list = sorted(df['TAG'].unique())
    selected_tag = st.selectbox("Select a Tree TAG to visualize its growth:", tree_tag_list)

    if selected_tag:
        tree_data = df[df['TAG'] == selected_tag]
        
        # Prepare data
        dbh_cols = ['D12', 'D15', 'D17', 'D19', 'Predicted_D21']
        available_cols = [c for c in dbh_cols if c in tree_data.columns]
        
        tree_growth_data = tree_data[available_cols].transpose()
        tree_growth_data.columns = ['DBH']
        
        year_map = {
            'D12': '2012', 'D15': '2015', 'D17': '2017', 
            'D19': '2019', 'Predicted_D21': '2021 (Pred)'
        }
        tree_growth_data['Year'] = [year_map.get(idx, idx) for idx in tree_growth_data.index]
        tree_growth_data = tree_growth_data.dropna()

        # Chart
        growth_chart = alt.Chart(tree_growth_data).mark_line(point=True).encode(
            x=alt.X('Year:N', sort=None, title='Census Year'),
            y=alt.Y('DBH', title='Diameter at Breast Height (cm)', scale=alt.Scale(zero=False)),
            tooltip=['Year', 'DBH']
        ).properties(
            title=f"Growth Trend for Tree TAG: {selected_tag}"
        ).interactive()
        
        st.altair_chart(growth_chart, use_container_width=True)
        
        # Details
        st.subheader("Tree Details")
        display_cols = ['TAG', 'SP', 'QUAD', 'Local_Density', 'Nearest_Neighbor_Dist', 'Predicted_Growth']
        display_cols = [c for c in display_cols if c in tree_data.columns]
        st.dataframe(tree_data[display_cols])

    # --- NAVIGATION BUTTONS ---
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    
    with col_nav1:
        if st.button("⬅️ Back to Spatial Map"):
            st.switch_page("pages/1_Spatial_Map.py")
            
    with col_nav3:
        if st.button("Go to Dashboard 🔄"):
            st.switch_page("pages/0_Dashboard.py")

else:
    st.warning("⚠️ Data not loaded. Please go to the **Home** page first.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")