import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_uber_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Uber Dataset Explorer")
st.write("Browse and analyze raw pickup GPS coordinates before running DBSCAN.")

if not os.path.exists(config.UBER_DATA_PATH):
    st.warning("Uber dataset not found. Please click below to download.")
    if st.button("🚀 Load Dataset"):
        load_uber_data()
        st.rerun()
else:
    df = pd.read_csv(config.UBER_DATA_PATH)
    
    st.subheader("Raw Coordinates Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latitude Distribution")
        fig_lat = px.histogram(df, x="Lat", nbins=30, title="Distribution of Pickups Latitude")
        st.plotly_chart(fig_lat, use_container_width=True)
        
    with col2:
        st.subheader("Longitude Distribution")
        fig_lon = px.histogram(df, x="Lon", nbins=30, title="Distribution of Pickups Longitude")
        st.plotly_chart(fig_lon, use_container_width=True)
        
    st.subheader("Raw Coordinates Scatter Map")
    st.write("This plot shows the raw coordinate spatial density prior to model clustering.")
    fig_scatter = px.scatter(df, x="Lon", y="Lat", opacity=0.4, title="Raw Taxi Pickups Map Projection")
    st.plotly_chart(fig_scatter, use_container_width=True)
