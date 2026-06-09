import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_turbofan_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Turbofan Telemetry Explorer")
st.write("Browse and analyze sensor telemetry and operational settings before running anomaly models.")

if not os.path.exists(config.TURBOFAN_DATA_PATH):
    st.warning("Turbofan dataset not found. Please click below to load.")
    if st.button("🚀 Load Dataset"):
        load_turbofan_data()
        st.rerun()
else:
    df = pd.read_csv(config.TURBOFAN_DATA_PATH)
    
    st.subheader("Raw Sensor Telemetry Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sensor Value Distributions")
        selected_sensor = st.selectbox("Select Sensor to Plot", config.SENSOR_COLS)
        fig_hist = px.histogram(df, x=selected_sensor, nbins=30, title=f"Distribution of {selected_sensor}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Operational Settings")
        selected_setting = st.selectbox("Select Setting to Plot", config.SETTING_COLS)
        fig_setting = px.histogram(df, x=selected_setting, nbins=30, title=f"Distribution of {selected_setting}")
        st.plotly_chart(fig_setting, use_container_width=True)
        
    st.subheader("Engine Lifetime Cycles")
    lifetimes = df.groupby("unit_number")["cycle"].max().reset_index()
    lifetimes.columns = ["Engine ID", "Cycles to Failure"]
    fig_life = px.bar(lifetimes, x="Engine ID", y="Cycles to Failure", title="Engine Operational Lifespans (Time-to-Failure)")
    st.plotly_chart(fig_life, use_container_width=True)
