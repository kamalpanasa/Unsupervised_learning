import streamlit as st
import os
import pickle
import pandas as pd
import sys

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.visualization.plots import plot_cluster_comparison_bar, plot_mood_scatter_3d

st.set_page_config(page_title="Results Explorer", layout="wide")

st.title("🔎 GMM Results Explorer")
st.write("Browse soft clusters distributions and lookup specific song mixture profiles.")

is_trained = os.path.exists(config.CLUSTERED_SPOTIFY_PATH)

if not is_trained:
    st.warning("Please train the model first on the Model Analysis page.")
else:
    df = pd.read_csv(config.CLUSTERED_SPOTIFY_PATH)
    
    with open(config.MODELS_PATH, "rb") as f:
        models = pickle.load(f)
    n_components = models["n_components"]
    
    # Calculate means
    cluster_means = df.groupby("cluster")[config.AUDIO_FEATURES].mean()
    
    tab_profiles, tab_soft, tab_lookup = st.tabs(["📊 Component Profiles", "🔍 Soft Membership Filter", "🔎 Song Mixture Lookup"])
    
    with tab_profiles:
        st.subheader("Component Attributes Profile Table")
        st.dataframe(cluster_means.style.format("{:.2f}"), use_container_width=True)
        
        fig_bar = plot_cluster_comparison_bar(cluster_means)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("3D Mood Space Projection")
        fig_3d = plot_mood_scatter_3d(df)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with tab_soft:
        st.subheader("Filter Songs by Soft Membership Certainty")
        st.write("Filter tracks based on GMM component probability threshold.")
        
        col_c, col_p = st.columns(2)
        with col_c:
            selected_comp = st.selectbox("Select Component", list(range(n_components)), format_func=lambda x: f"Component {x}")
        with col_p:
            min_prob = st.slider("Minimum Probability Threshold", min_value=0.10, max_value=1.00, value=0.60, step=0.05)
            
        prob_col = f"prob_cluster_{selected_comp}"
        filtered_df = df[df[prob_col] >= min_prob].sort_values(by=prob_col, ascending=False)
        
        st.write(f"Found **{len(filtered_df)}** songs matching the criteria.")
        st.dataframe(filtered_df[["track_name", "track_artist", "playlist_genre", prob_col] + config.AUDIO_FEATURES], use_container_width=True, hide_index=True)
        
    with tab_lookup:
        st.subheader("Lookup a Song's Mood Profile")
        st.write("Search for a song to check its probability breakdown.")
        
        search_query = st.text_input("Enter song name (e.g. 'bad guy', 'Shape of You'):", value="")
        if search_query:
            results = df[df["track_name"].str.contains(search_query, case=False, na=False)]
            if len(results) == 0:
                st.warning("No matches found.")
            else:
                selected_track = st.selectbox(
                    "Select the exact song:", 
                    results["track_name"].unique(),
                    format_func=lambda name: f"{name} by {results[results['track_name']==name]['track_artist'].values[0]}"
                )
                track_row = results[results["track_name"] == selected_track].iloc[0]
                
                # Format mixture probabilities
                prob_data = [{"GMM Component": f"Component {i}", "Probability": track_row[f"prob_cluster_{i}"]} for i in range(n_components)]
                prob_df = pd.DataFrame(prob_data)
                
                st.write(f"### {track_row['track_name']} — {track_row['track_artist']}")
                st.write(f"**Genre**: {track_row['playlist_genre'].upper()}")
                
                col_chart, col_feats = st.columns(2)
                with col_chart:
                    import plotly.express as px
                    fig_pie = px.pie(prob_df, names="GMM Component", values="Probability", title="Component Probability Distribution", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col_feats:
                    st.write("#### Audio Features:")
                    feats_df = pd.DataFrame({"Feature": config.AUDIO_FEATURES, "Value": [track_row[f] for f in config.AUDIO_FEATURES]})
                    st.dataframe(feats_df, use_container_width=True, hide_index=True)
