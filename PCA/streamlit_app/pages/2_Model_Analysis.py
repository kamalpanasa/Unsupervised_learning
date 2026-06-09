import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ PCA Model Analysis")
st.write("Fit a Principal Component Analysis mapping on superhero attributes.")

is_trained = os.path.exists(config.MODELS_PATH)

if st.button("🚀 Fit PCA Pipeline") or not is_trained:
    with st.spinner("Executing PCA fits..."):
        run_training_pipeline()
        st.success("PCA fitted successfully!")
        st.rerun()

with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)
    
st.header("PCA Pipeline Configuration")
st.write(f"- Fitted Principal Components: **3 (PC1, PC2, PC3)**")
st.write(f"- Scaler applied: **StandardScaler**")
st.write(f"- Processed attributes: {', '.join(config.STAT_COLS)}")
st.success("The PCA pipeline has run successfully and outputs are serialized.")
