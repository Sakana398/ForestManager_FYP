# pages/1_Spatial_Map.py
import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(page_title="ForestManager | Spatial Map", layout="wide")

st.title("🗺️ Spatial Map of Thinning Recommendations")

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    thinning_tags = set(st.session_state['df_thinning_recs']['TAG'])
    
    # --- NEW: Scenario Toggle ---
    view_mode = st.radio("View Mode:", ["Current Forest (Identify Targets)", "Post-Thinning Scenario"], horizontal=True)
    
    if view_mode == "Post-Thinning Scenario":
        # Filter OUT the thinned trees
        plot_data = df_all[~df_all['TAG'].isin(thinning_tags)]
        color_encoding = alt.value('green') # All remaining trees are green
        title = "Predicted Forest Structure After Thinning"
        st.success(f"Visualizing {len(plot_data)} remaining trees.")
    else:
        # Show ALL trees with status
        plot_data = df_all
        plot_data['Status'] = plot_data['TAG'].apply(lambda x: 'Remove' if x in thinning_tags else 'Keep')
        color_encoding = alt.Color('Status', scale=alt.Scale(domain=['Keep', 'Remove'], range=['lightgrey', 'red']))
        title = "Current Forest: Red dots indicate trees to remove"

    # Chart
    chart = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('XCO', title='X'),
        y=alt.Y('YCO', title='Y'),
        color=color_encoding,
        tooltip=['TAG', 'SP', 'Competition_Index', 'Predicted_Growth']
    ).properties(
        title=title,
        height=600
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
else:
    st.error("Please load data on the Home page first.")