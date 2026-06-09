import matplotlib.pyplot as plt
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram
from src.utils import config

def plot_dendrogram(linkage_matrix):
    """
    Renders a truncated linkage dendrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        show_contracted=True,
        ax=ax
    )
    ax.set_title("Pokémon Hierarchical Dendrogram (Last 30 Nodes Joined)")
    ax.set_xlabel("Node Size or Sample Index")
    ax.set_ylabel("Ward Linkage Distance")
    return fig

def plot_cluster_comparison_bar(cluster_means):
    """
    Creates a grouped bar chart comparing average base stats per cluster.
    """
    df_melted = cluster_means.reset_index().melt(
        id_vars="cluster", 
        var_name="Stat", 
        value_name="Average Value"
    )
    df_melted["cluster"] = df_melted["cluster"].astype(str)
    
    fig = px.bar(
        df_melted,
        x="Stat",
        y="Average Value",
        color="cluster",
        barmode="group",
        title="Comparison of Base Stats across Clusters"
    )
    return fig
