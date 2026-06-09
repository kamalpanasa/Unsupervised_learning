import streamlit as st
import os
import pickle
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.visualization.plots import plot_anomaly_score_trend, plot_sensor_telemetry, plot_multivariate_space_3d

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 Telemetry Anomaly Explorer")
st.write("Browse health profiles for specific engine units and view time-series anomaly highlights.")

is_trained = os.path.exists(config.CLUSTERED_TURBOFAN_PATH)

if not is_trained:
    st.warning("Please fit the Isolation Forest model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_TURBOFAN_PATH)
    
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    sensor_cols = models["sensor_cols"]
    
    st.sidebar.subheader("Select Target Engine")
    unit_ids = sorted(df["unit_number"].unique())
    selected_unit = st.sidebar.selectbox("Engine Unit ID", unit_ids, format_func=lambda x: f"Engine Unit {x}")
    
    unit_df = df[df["unit_number"] == selected_unit].copy()
    unit_df["Status"] = unit_df["anomaly"].map({0: "Normal", 1: "Degraded/Anomaly"})
    max_cycle = unit_df["cycle"].max()
    
    tab_health, tab_sensors = st.tabs(["📈 Engine Health History", "🎛️ Sensor Array Inspection"])
    
    with tab_health:
        st.subheader(f"Engine Unit {selected_unit} Health Curve")
        st.write(f"This engine was monitored for **{max_cycle} cycles** before failure. The chart below displays the decision score rising near its end-of-life.")
        fig_health = plot_anomaly_score_trend(unit_df)
        st.plotly_chart(fig_health, use_container_width=True)
        
        anom_count = unit_df["anomaly"].sum()
        anom_pct = (anom_count / len(unit_df)) * 100
        st.write(f"Flagged anomaly cycles: **{anom_count}** (**{anom_pct:.1f}%** of its lifetime).")
        
        st.write("#### Flagged Anomalous Cycles detail:")
        flagged_df = unit_df[unit_df["anomaly"] == 1][["cycle", "anomaly_score"] + sensor_cols[:4]]
        st.dataframe(flagged_df.style.format({"anomaly_score": "{:.3f}"}), use_container_width=True, hide_index=True)
        
    with tab_sensors:
        st.subheader("Sensor Value Line Charts")
        st.write("Red markers highlight cycles flagged as anomalies by the model.")
        selected_sensor = st.selectbox("Select Sensor Channel to Plot", sensor_cols)
        
        fig_sens = plot_sensor_telemetry(unit_df, selected_sensor)
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.subheader("Multivariate 3D Space")
        if len(sensor_cols) >= 3:
            fig_3d = plot_multivariate_space_3d(unit_df, sensor_cols[:3])
            st.plotly_chart(fig_3d, use_container_width=True)
