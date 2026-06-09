import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config

st.set_page_config(page_title="NASA Turbofan Anomaly Studio", layout="wide")

st.title("✈️ NASA Turbofan Engine Anomaly Studio")
st.write("Monitor aircraft engine sensor arrays and predict equipment degradation using Isolation Forests.")

is_trained = os.path.exists(config.CLUSTERED_TURBOFAN_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if os.path.exists(config.TURBOFAN_DATA_PATH):
        df_raw = pd.read_csv(config.TURBOFAN_DATA_PATH)
        st.metric("Engines Monitored", f"{len(df_raw['unit_number'].unique()):,}")
    else:
        st.metric("Engines Monitored", "Not Loaded")
with col2:
    if os.path.exists(config.TURBOFAN_DATA_PATH):
        st.metric("Total Logged Cycles", f"{len(df_raw):,}")
    else:
        st.metric("Total Logged Cycles", "Not Loaded")
with col3:
    if is_trained:
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.metric("Contamination Rate", f"{models['contamination']*100:.1f}%")
    else:
        st.metric("Contamination Rate", "Not Trained")
with col4:
    if is_trained:
        st.metric("Telemetry Features", f"{len(models['sensor_cols'])}")
    else:
        st.metric("Telemetry Features", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore raw sensor configurations and operational cycles.")
    st.write("* **Model Analysis**: Retrain Isolation Forests and adjust expected anomaly rates.")
    st.write("* **Results Explorer**: Inspect health profiles of specific engines and analyze sensor anomaly flags.")

with col_right:
    st.subheader("Active Sensors Configured")
    if is_trained:
        st.write(", ".join(models["sensor_cols"]))
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
