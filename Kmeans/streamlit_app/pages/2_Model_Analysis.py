import streamlit as st
import os
import pickle
import json
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline
from src.visualization.plots import plot_elbow_curve, plot_silhouette_scores

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ K-Means Model Configurator")
st.write("Tune cluster sizing, inspect elbow curves, and refit the customer segmentation model.")

is_trained = os.path.exists(config.MODELS_PATH) and os.path.exists(config.ELBOW_METRICS_PATH)

st.sidebar.subheader("Model Configuration")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    n_clusters_default = models.get("n_clusters", 5)
else:
    n_clusters_default = 5
    
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=n_clusters_default)

if st.sidebar.button("🚀 Train KMeans Model") or not is_trained:
    with st.spinner("Fitting K-Means clusters..."):
        run_training_pipeline(k=n_clusters)
        st.success("Model trained successfully!")
        st.rerun()

# Load elbow metrics
with open(config.ELBOW_METRICS_PATH, "r") as f:
    elbow_metrics = json.load(f)

tab_elbow, tab_silhouette = st.tabs(["📈 WCSS Elbow Curve", "📊 Silhouette Score"])

with tab_elbow:
    st.subheader("Within-Cluster Sum of Squares (Inertia)")
    st.write("The 'elbow' represents a point where adding more clusters yields diminishing returns in variance explanation.")
    fig_elbow = plot_elbow_curve(elbow_metrics["k_values"], elbow_metrics["wcss"], selected_k=n_clusters)
    st.plotly_chart(fig_elbow, use_container_width=True)

with tab_silhouette:
    st.subheader("Silhouette Score Analysis")
    st.write("Higher silhouette scores indicate better defined and separated clusters.")
    fig_sil = plot_silhouette_scores(elbow_metrics["k_values"], elbow_metrics["silhouette_scores"])
    st.plotly_chart(fig_sil, use_container_width=True)
