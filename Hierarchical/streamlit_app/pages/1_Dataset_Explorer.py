import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_raw_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Pokemon Dataset Explorer")
st.write("Browse and analyze base combat stats distributions before running clustering.")

if not os.path.exists(config.POKEMON_PATH):
    st.warning("Pokemon dataset not found. Please click below to load the dataset.")
    if st.button("🚀 Load Dataset"):
        load_raw_data()
        st.rerun()
else:
    df = pd.read_csv(config.POKEMON_PATH)
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attribute Distributions")
        selected_stat = st.selectbox("Select Attribute to Plot", config.STAT_COLS)
        fig_hist = px.histogram(df, x=selected_stat, nbins=35, title=f"Distribution of {selected_stat}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Combat Attribute Correlation")
        # Compute correlation on stats
        corr = df[config.STAT_COLS].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Base Stats"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.subheader("Top Stats Summary")
    st.dataframe(df[config.STAT_COLS].describe(), use_container_width=True)
