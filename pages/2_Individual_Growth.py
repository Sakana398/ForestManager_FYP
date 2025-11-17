# pages/2_Individual_Growth.py
import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(
    page_title="ForestManager | Growth Trends",
    layout="wide"
)

st.title("📈 Individual Tree Growth Trends")

# Check for FULL data from the Home page
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Allow user to select ANY tree, not just thinned ones
    tree_tag_list = sorted(df['TAG'].unique())
    selected_tag = st.selectbox("Select a Tree TAG to visualize its growth:", tree_tag_list)

    if selected_tag:
        tree_data = df[df['TAG'] == selected_tag]
        
        # Prepare data for plotting
        dbh_cols = ['D12', 'D15', 'D17', 'D19', 'Predicted_D21']
        tree_growth_data = tree_data[dbh_cols].transpose()
        tree_growth_data.columns = ['DBH']
        tree_growth_data['Year'] = ['2012', '2015', '2017', '2019', '2021 (Predicted)']
        
        # Clean up data for plotting
        tree_growth_data = tree_growth_data.dropna()

        # Line Chart
        growth_chart = alt.Chart(tree_growth_data).mark_line(point=True).encode(
            x=alt.X('Year:N', sort=None), # 'N' ensures categorical sorting (preserves order)
            y=alt.Y('DBH', title='Diameter at Breast Height (cm)', scale=alt.Scale(zero=False)),
            tooltip=['Year', 'DBH']
        ).properties(
            title=f"Growth Trend for Tree TAG: {selected_tag}"
        ).interactive()
        
        st.altair_chart(growth_chart, use_container_width=True)
        
        # Show the raw data for this tree
        st.subheader("Tree Details")
        st.dataframe(tree_data[['TAG', 'SP', 'QUAD', 'Local_Density', 'Nearest_Neighbor_Dist', 'Predicted_Growth']])
else:
    st.error("Data not loaded. Please go to the 'Home' page first.")