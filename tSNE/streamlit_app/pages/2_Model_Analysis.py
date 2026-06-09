import streamlit as st
import os
import pickle
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.train import run_training_pipeline

st.set_page_config(page_title="Model Analysis", layout="wide")

st.title("🎛️ t-SNE Model Configurator")
st.write("Tune perplexity and learning rate step sizes to optimize the non-linear embedding mapping.")

is_trained = os.path.exists(config.MODELS_PATH)

st.sidebar.subheader("t-SNE Parameters")
if is_trained:
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    perplexity_default = models.get("perplexity", 30)
    lr_default = models.get("learning_rate", "auto")
else:
    perplexity_default = 30
    lr_default = "auto"
    
perplexity = st.sidebar.slider("Perplexity", min_value=5, max_value=50, value=int(perplexity_default), step=5)
lr_option = st.sidebar.selectbox("Learning Rate", ["auto", 50, 100, 200, 500], index=0 if lr_default == "auto" else ["auto", 50, 100, 200, 500].index(lr_default))

actual_lr = lr_option if lr_option == "auto" else float(lr_option)

if st.sidebar.button("🚀 Re-compute t-SNE") or not is_trained:
    with st.spinner("Recomputing t-SNE coordinates..."):
        run_training_pipeline(perplexity=perplexity, learning_rate=actual_lr)
        st.success("t-SNE complete!")
        st.rerun()

# Display current configuration
with open(config.MODELS_PATH, "rb") as f:
    models = pickle.load(f)
    
st.header("t-SNE Model Details")
st.write(f"- Selected Perplexity: **{models['perplexity']}**")
st.write(f"- Selected Learning Rate: **{models['learning_rate']}**")
st.write("- Dimensionality space: **3D (tSNE 1, tSNE 2, tSNE 3)**")
st.write("- Features reduced: ")
st.write(", ".join(models["feature_cols"]))
