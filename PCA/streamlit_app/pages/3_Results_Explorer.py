import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.visualization.plots import plot_pca_projection_2d, plot_pca_projection_3d, plot_loadings_bar, plot_scree_plot

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 PCA Results Explorer")
st.write("Browse 2D/3D archetype projection coordinates and analyze loadings weights.")

is_trained = os.path.exists(config.CLUSTERED_HEROES_PATH)

if not is_trained:
    st.warning("Please fit the PCA model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_HEROES_PATH)
    
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
        
    stat_cols = models["stat_cols"]
    explained_var = models["explained_variance_ratio"]
    loadings = models["components"]
    
    # Sidebar filters
    st.sidebar.subheader("Visualization Settings")
    dim_selector = st.sidebar.radio("Projection Dimensions", ["2D (PC1 & PC2)", "3D (PC1, PC2 & PC3)"])
    
    alignments = df["Alignment"].dropna().unique()
    selected_alignments = st.sidebar.multiselect("Filter by Alignment", alignments, default=list(alignments))
    
    filtered_df = df[df["Alignment"].isin(selected_alignments)].copy()
    
    tab_proj, tab_loadings, tab_variance = st.tabs(["🔮 PCA Projections Map", "⚖️ Component Loadings", "📈 Explained Variance"])
    
    alignment_colors = {
        "good": "#1F77B4",
        "bad": "#D62728",
        "neutral": "#2CA02C",
        "empty": "#7F7F7F"
    }
    
    with tab_proj:
        st.subheader("Archetype Projection map")
        search_hero = st.text_input("Find and highlight a character:", value="")
        
        if search_hero:
            filtered_df["is_highlight"] = filtered_df["Name"].str.contains(search_hero, case=False, na=False)
            filtered_df["Marker Size"] = filtered_df["is_highlight"].map({True: 20, False: 6})
        else:
            filtered_df["Marker Size"] = 7
            
        if dim_selector == "2D (PC1 & PC2)":
            fig_proj = plot_pca_projection_2d(filtered_df, alignment_colors, search_hero)
            st.plotly_chart(fig_proj, use_container_width=True)
        else:
            fig_proj = plot_pca_projection_3d(filtered_df, alignment_colors, search_hero)
            st.plotly_chart(fig_proj, use_container_width=True)
            
    with tab_loadings:
        st.subheader("Attribute Loadings Weights")
        loadings_df = pd.DataFrame(loadings.T, index=stat_cols, columns=["PC1 Loading", "PC2 Loading", "PC3 Loading"])
        st.dataframe(loadings_df.style.format("{:.3f}"), use_container_width=True)
        
        loadings_melted = loadings_df.reset_index().rename(columns={"index": "Attribute"}).melt(id_vars="Attribute", var_name="Component", value_name="Weight")
        fig_loadings = plot_loadings_bar(loadings_melted)
        st.plotly_chart(fig_loadings, use_container_width=True)
        
    with tab_variance:
        st.subheader("Cumulative Explained Variance")
        cum_var = np.cumsum(explained_var)
        var_df = pd.DataFrame({
            "Component": ["PC1", "PC2", "PC3"],
            "Individual Variance": explained_var,
            "Cumulative Variance": cum_var
        })
        st.dataframe(var_df.style.format({"Individual Variance": "{:.2%}", "Cumulative Variance": "{:.2%}"}), use_container_width=True)
        
        fig_scree = plot_scree_plot(var_df)
        st.plotly_chart(fig_scree, use_container_width=True)
