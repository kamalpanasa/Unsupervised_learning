import streamlit as st
import os
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import config
from src.modeling.analysis import get_cluster_stats, get_cluster_profiles

st.set_page_config(page_title="Pokémon Clustering Studio", layout="wide")

st.title("🐾 Pokémon Hierarchical Clustering Studio")
st.write("Group Pokémon into distinct combat archetypes using Agglomerative Clustering.")

is_trained = os.path.exists(config.CLUSTERED_POKEMON_PATH) and os.path.exists(config.MODELS_PATH)

st.header("Database Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if os.path.exists(config.POKEMON_PATH):
        df_raw = pd.read_csv(config.POKEMON_PATH)
        st.metric("Total Pokemon in Catalog", f"{len(df_raw):,}")
    else:
        st.metric("Total Pokemon in Catalog", "Not Loaded")
with col2:
    if is_trained:
        df_clustered = pd.read_csv(config.CLUSTERED_POKEMON_PATH)
        st.metric("Clustered Records", f"{len(df_clustered):,}")
    else:
        st.metric("Clustered Records", "Not Trained")
with col3:
    if is_trained:
        stats = get_cluster_stats(df_clustered)
        st.metric("Number of Segments (K)", f"{len(stats)}")
    else:
        st.metric("Number of Segments (K)", "N/A")
with col4:
    # Check if Pikachu was renamed
    if is_trained:
        has_vilohih = "vilohih" in df_clustered["Name"].values
        st.metric("Vilohih Renamed Status", "Active" if has_vilohih else "Inactive")
    else:
        st.metric("Vilohih Renamed Status", "N/A")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Application Pages")
    st.write("* **Dataset Explorer**: Explore raw combat stats and correlations.")
    st.write("* **Model Analysis**: Retrain models and explore the Dendrogram Tree.")
    st.write("* **Results Explorer**: Profile clusters and browse individual Pokémon.")

with col_right:
    st.subheader("Discovered Archetypes Preview")
    if is_trained:
        stats = get_cluster_stats(df_clustered)
        profiles = get_cluster_profiles(stats)
        
        for cid, prof in profiles.items():
            st.write(f"**Cluster {cid}: {prof['name']}** ({prof['count']} Pokémon)")
            st.write(f"- Key Characteristic: High {prof['top_stat']}")
            st.write(f"- Summary: {prof['description']}")
            st.write("")
    else:
        st.warning("The model has not been trained yet. Navigate to the **Model Analysis** page to fit the clustering pipeline.")
