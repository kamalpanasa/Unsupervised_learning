import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ Isolation Forest Configurator")
st.write("Tune contamination rates and fit Isolation Forest anomaly classifiers.")

is_trained = os.path.exists(config.MODELS_PATH)

st.sidebar.subheader("Model Configuration")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    contamination_default = models.get("contamination", 0.10)
else:
    contamination_default = 0.10
    
contamination = st.sidebar.slider("Contamination Rate (Expected Anomaly %)", min_value=0.01, max_value=0.25, value=float(contamination_default), step=0.01)

if st.sidebar.button("🚀 Train Isolation Forest") or not is_trained:
    with st.spinner("Refitting Isolation Forest..."):
        run_training_pipeline(contamination=contamination)
        st.success("Isolation Forest fitted successfully!")
        st.rerun()

# Display GMM parameters
with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)
    
st.header("Isolation Forest Model Status")
st.write(f"- Fitted Contamination Threshold: **{models['contamination']*100:.1f}%**")
st.write(f"- Target Sensors Modeled: **{len(models['sensor_cols'])}**")
st.write(", ".join(models["sensor_cols"]))
st.success("Telemetry variables scaled and model successfully saved.")
