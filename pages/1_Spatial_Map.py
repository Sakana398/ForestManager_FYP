# pages/1_Spatial_Map.py
import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(page_title="ForestManager | Spatial Map", layout="wide")

st.title("🗺️ Spatial Map of Thinning Recommendations")

if 'df' in st.session_state and 'df_thinning_recs' in st.session_state:
    df_all = st.session_state['df'].copy()
    thinning_tags = set(st.session_state['df_thinning_recs']['TAG'])
    
    # --- Scenario Toggle ---
    view_mode = st.radio(
        "View Mode:", 
        ["Current Forest (Identify Targets)", "Post-Thinning Scenario"], 
        horizontal=True
    )
    
    if view_mode == "Post-Thinning Scenario":
        # Filter OUT the thinned trees to show only what remains
        plot_data = df_all[~df_all['TAG'].isin(thinning_tags)].copy()
        color_encoding = alt.value('green')
        title = "Predicted Forest Structure After Thinning"
        st.success(f"Visualizing {len(plot_data)} remaining trees.")
    else:
        # Show ALL trees with status colored
        plot_data = df_all.copy()
        plot_data['Status'] = plot_data['TAG'].apply(lambda x: 'Remove' if x in thinning_tags else 'Keep')
        color_encoding = alt.Color('Status', scale=alt.Scale(domain=['Keep', 'Remove'], range=['lightgrey', 'red']))
        title = "Current Forest: Red dots indicate trees to remove"

    # Altair Chart
    chart = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('XCO', title='X Coordinate'),
        y=alt.Y('YCO', title='Y Coordinate'),
        color=color_encoding,
        tooltip=['TAG', 'SP', 'Competition_Index', 'Predicted_Growth', 'D19']
    ).properties(
        title=title,
        height=600
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # --- NAVIGATION BUTTONS ---
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    
    with col_nav1:
        if st.button("⬅️ Back to Dashboard"):
            st.switch_page("pages/0_Dashboard.py")
            
    with col_nav3:
        if st.button("Go to Individual Growth ➡️"):
            st.switch_page("pages/2_Individual_Growth.py")
    
else:
    st.warning("⚠️ No data found. Please go to the **Home** page and load the dashboard first.")
    if st.button("Go to Home"):
        st.switch_page("ForestManager_app.py")