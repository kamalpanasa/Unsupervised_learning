import plotly.express as px
import pandas as pd

def plot_elbow_curve(k_values, wcss, selected_k=5):
    """
    WCSS Line chart (Elbow).
    """
    fig = px.line(
        x=k_values,
        y=wcss,
        markers=True,
        title="Elbow Curve (WCSS vs K)",
        labels={"x": "Number of Clusters (K)", "y": "WCSS (Inertia)"}
    )
    return fig

def plot_silhouette_scores(k_values, scores):
    """
    Silhouette Score Bar chart.
    """
    fig = px.bar(
        x=k_values,
        y=scores,
        title="Silhouette Score vs K",
        labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"}
    )
    return fig

def plot_customer_scatter_3d(df):
    """
    Renders 3D customer clusters.
    """
    df_plot = df.copy()
    df_plot["Segment"] = df_plot["cluster"].apply(lambda c: f"Cluster {c}")
    
    fig = px.scatter_3d(
        df_plot,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color="Segment",
        hover_data=["CustomerID", "Gender"],
        title="3D Customer Clusters Projection Space",
        opacity=0.8
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_cluster_comparison_bar(cluster_means):
    """
    Grouped bar comparing variables.
    """
    df_melted = cluster_means.reset_index().melt(
        id_vars="cluster", 
        var_name="Feature", 
        value_name="Average Value"
    )
    df_melted["cluster"] = df_melted["cluster"].apply(lambda c: f"Cluster {c}")
    
    fig = px.bar(
        df_melted,
        x="Feature",
        y="Average Value",
        color="cluster",
        barmode="group",
        title="Average Profile Features comparison"
    )
    return fig
