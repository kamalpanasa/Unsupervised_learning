import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline
from src.visualization.plots import plot_dendrogram

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🌳 Hierarchical Model Analysis")
st.write("Tune cluster settings, re-run Agglomerative Clustering, and view linkage hierarchies.")

# Check if trained
is_trained = os.path.exists(config.MODELS_PATH)

st.sidebar.subheader("Hyperparameter Settings")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    n_clusters_default = models.get("n_clusters", 5)
else:
    n_clusters_default = 5
    
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=n_clusters_default)

if st.sidebar.button("🚀 Run Clustering Pipeline") or not is_trained:
    with st.spinner("Training clustering model..."):
        run_training_pipeline(n_clusters=n_clusters)
        st.success("Agglomerative Clustering complete!")
        st.rerun()

# Reload model
with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)

linkage_matrix = models["linkage_matrix"]

st.header("Agglomerative Hierarchy (Dendrogram)")
st.write("The dendrogram illustrates how the algorithm groups Pokémon bottom-up by joining similar clusters.")

fig_dendro = plot_dendrogram(linkage_matrix)
st.pyplot(fig_dendro)
