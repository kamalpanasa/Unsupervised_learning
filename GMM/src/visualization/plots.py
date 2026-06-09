import plotly.express as px
import pandas as pd

def plot_cluster_comparison_bar(cluster_means):
    """
    Plots a grouped bar chart of audio features across clusters.
    """
    df_melted = cluster_means.reset_index().melt(
        id_vars="cluster", 
        var_name="Feature", 
        value_name="Average Value"
    )
    df_melted["cluster"] = df_melted["cluster"].apply(lambda c: f"Component {c}")
    
    fig = px.bar(
        df_melted,
        x="Feature",
        y="Average Value",
        color="cluster",
        barmode="group",
        title="Comparison of Audio Features across GMM Components"
    )
    return fig

def plot_mood_scatter_3d(df):
    """
    Projects songs in 3D audio space (danceability, energy, valence).
    """
    df_plot = df.copy()
    df_plot["Dominant Component"] = df_plot["cluster"].apply(lambda c: f"Component {c}")
    
    # Sample if too large
    if len(df_plot) > 1500:
        df_plot = df_plot.sample(1500, random_state=42)
        
    fig = px.scatter_3d(
        df_plot,
        x="danceability",
        y="energy",
        z="valence",
        color="Dominant Component",
        hover_data=["track_name", "track_artist", "playlist_genre"],
        title="Danceability vs Energy vs Valence (Colored by Dominant Component)",
        opacity=0.6
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig
