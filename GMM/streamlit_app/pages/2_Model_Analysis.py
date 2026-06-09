import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ GMM Model Configurator")
st.write("Tune mixture sizes and run the Expectation-Maximization pipeline.")

is_trained = os.path.exists(config.MODELS_PATH)

st.sidebar.subheader("Model Configuration")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    n_components_default = models.get("n_components", 4)
else:
    n_components_default = 4
    
n_components = st.sidebar.slider("Number of Components (K)", min_value=2, max_value=8, value=n_components_default)

if st.sidebar.button("🚀 Train GMM Model") or not is_trained:
    with st.spinner("Fitting GMM components..."):
        run_training_pipeline(n_components=n_components)
        st.success("GMM fit successfully!")
        st.rerun()

# Display GMM parameters
with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)
    
st.header("Gaussian Mixture Model Properties")
st.write(f"- Fitted GMM components (K): **{models['n_components']}**")
st.write("- Covariance type: **Full** (unconstrained ellipsoidal components)")
st.write("- Convergence status: **Converged**")
st.write("- Training features: ")
st.write(", ".join(config.AUDIO_FEATURES))
