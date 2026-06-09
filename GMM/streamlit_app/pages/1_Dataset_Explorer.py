import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_spotify_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Spotify Audio Features Explorer")
st.write("Browse and analyze audio metrics distribution before running the Gaussian Mixture Model.")

if not os.path.exists(config.SPOTIFY_DATA_PATH):
    st.warning("Spotify dataset not found. Please click below to download.")
    if st.button("🚀 Load Dataset"):
        load_spotify_data()
        st.rerun()
else:
    df = pd.read_csv(config.SPOTIFY_DATA_PATH)
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Values Distribution")
        selected_feature = st.selectbox("Select Feature to Plot", config.AUDIO_FEATURES)
        fig_hist = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Genre Composition")
        genre_counts = df["playlist_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig_pie = px.pie(genre_counts, names="Genre", values="Count", title="Distribution of Playlist Genres")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.subheader("Correlation Heatmap")
    corr = df[config.AUDIO_FEATURES].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Audio Feature Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)
