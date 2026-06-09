import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config

st.set_page_config(page_title="Spotify GMM Mood Profiler", layout="wide")

st.title("🎵 Spotify Mood Profiler (GMM Studio)")
st.write("Understand song profiles as continuous mixture distributions (soft clusters) using Gaussian Mixture Models.")

is_trained = os.path.exists(config.CLUSTERED_SPOTIFY_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(config.SPOTIFY_DATA_PATH):
        df_raw = pd.read_csv(config.SPOTIFY_DATA_PATH)
        st.metric("Total Tracks Sampled", f"{len(df_raw):,}")
    else:
        st.metric("Total Tracks Sampled", "Not Loaded")
with col2:
    if is_trained:
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.metric("Active GMM Components (K)", f"{models['n_components']}")
    else:
        st.metric("Active GMM Components (K)", "Not Trained")
with col3:
    if is_trained:
        df_clustered = pd.read_csv(config.CLUSTERED_SPOTIFY_PATH)
        st.metric("Audio Features Modeled", f"{len(config.AUDIO_FEATURES)}")
    else:
        st.metric("Audio Features Modeled", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore distributions of danceability, energy, tempo, and other metrics.")
    st.write("* **Model Analysis**: Retrain GMM model with different components (K).")
    st.write("* **Results Explorer**: Multi-dimensional GMM projections, soft probability filters, and individual song search.")

with col_right:
    st.subheader("Gaussian Components Configuration")
    if is_trained:
        st.write(f"**Current number of components**: {models['n_components']}")
        st.write(f"**Model Features**: {', '.join(config.AUDIO_FEATURES)}")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
