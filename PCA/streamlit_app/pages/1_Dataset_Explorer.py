import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_heroes_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Superhero Power Stats Explorer")
st.write("Browse raw attributes distributions and class alignments before executing PCA.")

if not os.path.exists(config.HEROES_DATA_PATH):
    st.warning("Superhero dataset not found. Click below to load data.")
    if st.button("🚀 Load Dataset"):
        load_heroes_data()
        st.rerun()
else:
    df = pd.read_csv(config.HEROES_DATA_PATH)
    
    st.subheader("Raw Characters Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attribute Value Distribution")
        selected_stat = st.selectbox("Select Attribute to Plot", config.STAT_COLS)
        fig_hist = px.histogram(df, x=selected_stat, nbins=30, title=f"Distribution of {selected_stat}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Alignment Composition")
        align_counts = df["Alignment"].fillna("unknown").str.lower().value_counts().reset_index()
        align_counts.columns = ["Alignment", "Count"]
        fig_pie = px.pie(align_counts, names="Alignment", values="Count", title="Distribution of Character Alignments")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.subheader("Feature Correlation Heatmap")
    corr = df[config.STAT_COLS].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Base Stats Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)
