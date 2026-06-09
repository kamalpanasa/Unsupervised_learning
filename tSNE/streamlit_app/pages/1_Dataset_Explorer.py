import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_anime_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Anime Face Dataset Explorer")
st.write("Browse and analyze continuous geometry variables before running t-SNE.")

if not os.path.exists(config.ANIME_DATA_PATH):
    st.warning("Anime dataset not found. Please click below to generate.")
    if st.button("🚀 Load Dataset"):
        load_anime_data()
        st.rerun()
else:
    df = pd.read_csv(config.ANIME_DATA_PATH)
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Face Geometry Distributions")
        selected_geom = st.selectbox("Select Geometry Feature to Plot", config.GEOM_COLS)
        fig_hist = px.histogram(df, x=selected_geom, nbins=30, title=f"Distribution of {selected_geom}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Style Group Composition")
        style_counts = df["Style_Group"].value_counts().reset_index()
        style_counts.columns = ["Style Group", "Count"]
        fig_pie = px.pie(style_counts, names="Style Group", values="Count", title="Distribution of Character Style Groups")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.subheader("Geometry Features Correlation Matrix")
    corr = df[config.GEOM_COLS].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Base Geometries Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)
