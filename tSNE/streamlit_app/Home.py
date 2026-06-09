import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config

st.set_page_config(page_title="Anime Face t-SNE Studio", layout="wide")

st.title("🎭 Anime Face t-SNE Studio")
st.write("Map high-dimensional anime face descriptors down to 2D/3D non-linear neighborhood configurations.")

is_trained = os.path.exists(config.CLUSTERED_ANIME_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(config.ANIME_DATA_PATH):
        df_raw = pd.read_csv(config.ANIME_DATA_PATH)
        st.metric("Total Characters Logged", f"{len(df_raw):,}")
    else:
        st.metric("Total Characters Logged", "Not Loaded")
with col2:
    if is_trained:
        st.metric("Active t-SNE Dimensions", "3 (tSNE_1, tSNE_2, tSNE_3)")
    else:
        st.metric("Active t-SNE Dimensions", "Not Trained")
with col3:
    if is_trained:
        st.metric("Archetype Classes", "5 Style Groups")
    else:
        st.metric("Archetype Classes", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore distributions of continuous facial geometry variables.")
    st.write("* **Model Analysis**: Tune perplexity and learning rate to recompute non-linear projections.")
    st.write("* **Results Explorer**: 2D/3D neighborhood scatters and style cluster attributes browser.")

with col_right:
    st.subheader("t-SNE Parameters Configuration")
    if is_trained:
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.write(f"**Current Perplexity**: {models['perplexity']}")
        st.write(f"**Current Learning Rate**: {models['learning_rate']}")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
