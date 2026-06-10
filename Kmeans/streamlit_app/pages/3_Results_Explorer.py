import streamlit as st
import os
import pickle
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.analysis import get_cluster_stats, get_cluster_personas
from src.visualization.plots import plot_customer_scatter_3d, plot_cluster_comparison_bar

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 Customer Segments Results Explorer")
st.write("Browse 3D projection mappings and analyze the properties of shopper archetypes.")

is_trained = os.path.exists(config.CLUSTERED_MALL_PATH)

if not is_trained:
    st.warning("Please fit the model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_MALL_PATH)
    
    # Calculate stats
    stats = get_cluster_stats(df)
    personas = get_cluster_personas(stats)
    
    tab_stats, tab_3d, tab_browse = st.tabs(["📊 Segment Statistics", "🔮 3D Cluster Projection", "🔍 Customer Browser"])
    
    with tab_stats:
        st.subheader("Average Features per Cluster")
        st.dataframe(stats.style.format({col: "{:.1f}" for col in config.NUMERICAL_COLS}), use_container_width=True)
        
        st.write("#### Profile Feature Distributions")
        fig_bar = plot_cluster_comparison_bar(stats[config.NUMERICAL_COLS])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab_3d:
        st.subheader("3D Customer Space")
        fig_3d = plot_customer_scatter_3d(df)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with tab_browse:
        st.subheader("Browse Shoppers by Segment")
        cluster_ids = list(stats.index)
        selected_cid = st.selectbox(
            "Select Customer Segment ID", 
            cluster_ids, 
            format_func=lambda x: f"Cluster {x} - {personas[x]['name']} ({personas[x]['count']} members)"
        )
        
        prof = personas[selected_cid]
        st.info(f"**Customer Profile**: {prof['description']}")
        
        cluster_df = df[df["cluster"] == selected_cid]
        st.dataframe(cluster_df[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]], use_container_width=True, hide_index=True)
