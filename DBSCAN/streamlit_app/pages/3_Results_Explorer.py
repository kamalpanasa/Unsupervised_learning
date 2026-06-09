import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.visualization.plots import plot_coordinate_map, plot_cluster_volumes_bar

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 Hotspot Results Explorer")
st.write("Examine coordinate density hotspots and isolate outlying trips.")

is_trained = os.path.exists(config.CLUSTERED_UBER_PATH)

if not is_trained:
    st.warning("Please train the model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_UBER_PATH)
    
    # Format labels for discrete colors
    df["cluster_label"] = df["cluster"].apply(lambda c: "Noise/Outliers" if c == -1 else f"Hotspot {c}")
    
    # Custom color map keeping Noise gray
    unique_labels = sorted(df["cluster_label"].unique())
    color_map = {}
    for lbl in unique_labels:
        if lbl == "Noise/Outliers":
            color_map[lbl] = "#BDC3C7"  # Gray
        else:
            color_map[lbl] = px.colors.qualitative.Plotly[hash(lbl) % len(px.colors.qualitative.Plotly)]
            
    tab_map, tab_analysis = st.tabs(["📍 NYC Hotspot Map", "📊 Hotspot Breakdown"])
    
    with tab_map:
        st.subheader("Coordinates Scatter Map")
        st.write("Hover over points to see GPS coordinates. Noise points are highlighted in gray.")
        fig_map = plot_coordinate_map(df, color_map)
        st.plotly_chart(fig_map, use_container_width=True)
        
    with tab_analysis:
        st.subheader("Trip Distribution per Hotspot")
        counts = df["cluster_label"].value_counts().reset_index()
        counts.columns = ["Cluster Group", "Trip Count"]
        
        st.dataframe(counts, use_container_width=True, hide_index=True)
        
        fig_bar = plot_cluster_volumes_bar(counts, color_map)
        st.plotly_chart(fig_bar, use_container_width=True)
