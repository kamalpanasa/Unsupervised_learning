import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config

st.set_page_config(page_title="Uber DBSCAN Studio", layout="wide")

st.title("🚖 Uber NYC pickup DBSCAN Studio")
st.write("Isolate outliers and extract high-density passenger pickup zones from coordinates.")

is_trained = os.path.exists(config.CLUSTERED_UBER_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if os.path.exists(config.UBER_DATA_PATH):
        df_raw = pd.read_csv(config.UBER_DATA_PATH)
        st.metric("Total Trips Analysed", f"{len(df_raw):,}")
    else:
        st.metric("Total Trips Analysed", "Not Loaded")
with col2:
    if is_trained:
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.metric("Detected Hotspots", f"{models['n_clusters']}")
    else:
        st.metric("Detected Hotspots", "Not Trained")
with col3:
    if is_trained:
        st.metric("Noise Points (Outliers)", f"{models['n_noise']}")
    else:
        st.metric("Noise Points (Outliers)", "N/A")
with col4:
    if is_trained:
        noise_pct = (models["n_noise"] / len(df_raw)) * 100
        st.metric("Outliers Rate", f"{noise_pct:.1f}%")
    else:
        st.metric("Outliers Rate", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Scatter maps showing raw trip distribution coordinates.")
    st.write("* **Model Analysis**: retrain DBSCAN model and configure neighborhood densities.")
    st.write("* **Results Explorer**: Color-coded hot zones and outliers breakdown.")

with col_right:
    st.subheader("DBSCAN Parameters Configuration")
    if is_trained:
        st.write(f"**Current neighborhood size (eps)**: {models['eps']}")
        st.write(f"**Current density weight (min_samples)**: {models['min_samples']}")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
