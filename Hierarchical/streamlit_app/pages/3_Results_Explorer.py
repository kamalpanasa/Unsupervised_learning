import streamlit as st
import os
import pickle
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.modeling.analysis import get_cluster_stats, get_cluster_profiles
from src.visualization.plots import plot_cluster_comparison_bar

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 Cluster Results Explorer")
st.write("Browse and analyze the properties of each cluster group.")

is_trained = os.path.exists(config.CLUSTERED_POKEMON_PATH)

if not is_trained:
    st.warning("Please train the model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_POKEMON_PATH)
    
    # Calculate stats
    stats = get_cluster_stats(df)
    profiles = get_cluster_profiles(stats)
    
    tab_stats, tab_browse, tab_search = st.tabs(["📊 Segment Statistics", "🔍 Archetype Browser", "🔎 Pokémon Lookup"])
    
    with tab_stats:
        st.subheader("Average Combat Stats per Cluster")
        st.dataframe(stats.style.format({col: "{:.1f}" for col in config.STAT_COLS}), use_container_width=True)
        
        st.write("#### Feature Distribution by Segment")
        fig_bar = plot_cluster_comparison_bar(stats[config.STAT_COLS])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab_browse:
        st.subheader("Browse Pokémon by Archetype")
        cluster_ids = list(stats.index)
        selected_cid = st.selectbox("Select Segment ID", cluster_ids, format_func=lambda x: f"Cluster {x} - {profiles[x]['name']} ({profiles[x]['count']} members)")
        
        prof = profiles[selected_cid]
        st.info(f"**Description**: {prof['description']}")
        
        cluster_df = df[df["cluster"] == selected_cid]
        st.dataframe(cluster_df[["Name", "Type 1", "Type 2"] + config.STAT_COLS], use_container_width=True, hide_index=True)
        
    with tab_search:
        st.subheader("Look up a specific Pokémon")
        st.write("Search for any Pokémon name (e.g. 'vilohih' to inspect Pikachu's renamed segment status).")
        
        search_query = st.text_input("Enter Pokemon Name:", value="")
        if search_query:
            matches = df[df["Name"].str.contains(search_query, case=False, na=False)]
            if len(matches) == 0:
                st.warning("No matches found.")
            else:
                st.dataframe(matches[["Name", "Type 1", "Type 2", "cluster"] + config.STAT_COLS], use_container_width=True, hide_index=True)
                
                # If there is a match, explain its segment membership
                selected_match = st.selectbox("Select to show profile:", matches["Name"].unique())
                row = matches[matches["Name"] == selected_match].iloc[0]
                cid = int(row["cluster"])
                st.write(f"**{selected_match}** is grouped into **Cluster {cid}** (*{profiles[cid]['name']}*).")
                st.write(f"- {profiles[cid]['description']}")
