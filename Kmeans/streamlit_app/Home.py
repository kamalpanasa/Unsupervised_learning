import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config
from src.modeling.analysis import get_cluster_stats, get_cluster_personas

st.set_page_config(page_title="Customer Segmentation Studio", layout="wide")

st.title("🛍️ Mall Customer Segmentation Studio (K-Means)")
st.write("Group retail customers into demographic spending personas based on Age, Income, and Spending Score.")

is_trained = os.path.exists(config.CLUSTERED_MALL_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(config.MALL_DATA_PATH):
        df_raw = pd.read_csv(config.MALL_DATA_PATH)
        st.metric("Total Customers Cataloged", f"{len(df_raw):,}")
    else:
        st.metric("Total Customers Cataloged", "Not Loaded")
with col2:
    if is_trained:
        df_clustered = pd.read_csv(config.CLUSTERED_MALL_PATH)
        import pickle
        with open(config.MODELS_PATH, "rb") as f:
            models = pickle.load(f)
        st.metric("Customer Segments (K)", f"{models['n_clusters']}")
    else:
        st.metric("Customer Segments (K)", "Not Trained")
with col3:
    if is_trained:
        st.metric("Features Modeled", "3 (Age, Income, Spending)")
    else:
        st.metric("Features Modeled", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore distributions of customer age, gender, and spending.")
    st.write("* **Model Analysis**: Tune KMeans cluster settings (K) and view Elbow validation metrics.")
    st.write("* **Results Explorer**: 3D projection spaces, customer profiles, and segment listings.")

with col_right:
    st.subheader("Retail Customer Personas Preview")
    if is_trained:
        stats = get_cluster_stats(df_clustered)
        personas = get_cluster_personas(stats)
        
        for cid, prof in personas.items():
            st.write(f"**Cluster {cid}: {prof['name']}** ({prof['count']} customers)")
            st.write(f"- Average Age: {prof['avg_age']:.1f} years")
            st.write(f"- Profile: {prof['description']}")
            st.write("")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to run the training pipeline.")
