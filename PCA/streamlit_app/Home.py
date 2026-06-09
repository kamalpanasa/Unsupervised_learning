import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config

st.set_page_config(page_title="Superhero PCA Studio", layout="wide")

st.title("🦸‍♂️ Superhero PCA Studio")
st.write("Deconstruct and map high-dimensional superhero power stats to visual archetype dimensions.")

is_trained = os.path.exists(config.CLUSTERED_HEROES_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(config.HEROES_DATA_PATH):
        df_raw = pd.read_csv(config.HEROES_DATA_PATH)
        st.metric("Total Superheroes Cataloged", f"{len(df_raw):,}")
    else:
        st.metric("Total Superheroes Cataloged", "Not Loaded")
with col2:
    if is_trained:
        st.metric("Principal Components Modeled", "3 (PC1, PC2, PC3)")
    else:
        st.metric("Principal Components Modeled", "Not Trained")
with col3:
    if is_trained:
        st.metric("Mapped Combat Attributes", f"{len(config.STAT_COLS)}")
    else:
        st.metric("Mapped Combat Attributes", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore raw distributions, alignment classes, and correlation maps.")
    st.write("* **Model Analysis**: Fit the PCA dimensionality reduction pipeline.")
    st.write("* **Results Explorer**: 2D/3D projection plots, Scree plots, and loadings weights analysis.")

with col_right:
    st.subheader("PCA Features Map")
    if is_trained:
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.write(f"**Principal Components**: PC1, PC2, PC3")
        st.write(f"**Attributes reduced**: {', '.join(config.STAT_COLS)}")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
