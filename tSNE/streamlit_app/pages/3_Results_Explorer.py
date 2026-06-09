import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.visualization.plots import plot_tsne_2d, plot_tsne_3d

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 t-SNE Results Explorer")
st.write("Browse 2D/3D non-linear maps and inspect style archetype attributes.")

is_trained = os.path.exists(config.CLUSTERED_ANIME_PATH)

if not is_trained:
    st.warning("Please run t-SNE first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_ANIME_PATH)
    
    # Sidebar
    st.sidebar.subheader("Visualization Settings")
    dim_selector = st.sidebar.radio("Map Dimension", ["2D (tSNE 1 & tSNE 2)", "3D (tSNE 1, tSNE 2 & tSNE 3)"])
    
    style_colors = {
        "Chibi": "#FF69B4",
        "Kuudere": "#4169E1",
        "Shonen Protagonist": "#FF8C00",
        "Tsundere": "#FF4500",
        "Villain": "#8A2BE2"
    }
    
    tab_proj, tab_explore = st.tabs(["🔮 t-SNE Projection Map", "🔍 Style Group Attributes"])
    
    with tab_proj:
        st.subheader("Anime Face Style Projection")
        st.write("t-SNE preserves local neighborhoods, placing similar styles together.")
        
        if dim_selector == "2D (tSNE 1 & tSNE 2)":
            fig_tsne = plot_tsne_2d(df, style_colors)
            st.plotly_chart(fig_tsne, use_container_width=True)
        else:
            fig_tsne = plot_tsne_3d(df, style_colors)
            st.plotly_chart(fig_tsne, use_container_width=True)
            
    with tab_explore:
        st.subheader("Face Geometry Statistics by Archetype")
        selected_style = st.selectbox("Select Style Archetype", list(style_colors.keys()))
        style_df = df[df["Style_Group"] == selected_style]
        
        st.write(f"#### Average Geometry for {selected_style}:")
        st.dataframe(style_df[config.GEOM_COLS].mean().to_frame("Average Value").T.style.format("{:.2f}"), use_container_width=True)
        
        st.write("#### Character Samples:")
        st.dataframe(style_df[["Character_ID", "Hair_Color", "Eye_Color"] + config.GEOM_COLS], use_container_width=True, hide_index=True)
