import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ DBSCAN Model Analysis")
st.write("Tune neighborhood boundaries and cluster densities to extract spatial hotspots.")

is_trained = os.path.exists(config.MODELS_PATH)

st.sidebar.subheader("Model Configuration")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    eps_default = models.get("eps", 0.15)
    min_samples_default = models.get("min_samples", 10)
else:
    eps_default = 0.15
    min_samples_default = 10
    
eps = st.sidebar.slider("Neighborhood Size (eps)", min_value=0.05, max_value=1.0, value=eps_default, step=0.05)
min_samples = st.sidebar.slider("Min Samples", min_value=2, max_value=50, value=min_samples_default, step=1)

if st.sidebar.button("🚀 Fit DBSCAN Model") or not is_trained:
    with st.spinner("Refitting DBSCAN..."):
        run_training_pipeline(eps=eps, min_samples=min_samples)
        st.success("DBSCAN fitted successfully!")
        st.rerun()

# Display current model properties
with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)
    
st.header("Current Model Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("Detected Hotspots", models["n_clusters"])
with col2:
    st.metric("Noise/Outliers", models["n_noise"])
    
st.markdown("""
### DBSCAN Parameters Explained:
1. **eps (epsilon)**: The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
   - A larger `eps` will merge nearby clusters and reduce noise flags.
   - A smaller `eps` will isolate small tight clusters but mark more points as outliers.
2. **min_samples**: The number of samples in a neighborhood for a point to be considered a core point.
   - Increasing this requires clusters to be denser to form, flagging more border points as noise.
""")
