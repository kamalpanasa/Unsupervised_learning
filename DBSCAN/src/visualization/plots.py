import plotly.express as px

def plot_coordinate_map(df, color_map):
    """
    Renders coordinates scatter map.
    """
    fig = px.scatter(
        df,
        x="Lon",
        y="Lat",
        color="cluster_label",
        color_discrete_map=color_map,
        labels={"Lon": "Longitude", "Lat": "Latitude", "cluster_label": "Cluster Group"},
        title="DBSCAN Hotspots & Noise Projection",
        hover_data=["Lat", "Lon"]
    )
    fig.update_layout(
        font=dict(family="Source Sans Pro, sans-serif", size=12),
        xaxis=dict(title="Longitude"),
        yaxis=dict(title="Latitude")
    )
    return fig

def plot_cluster_volumes_bar(counts, color_map):
    """
    Plots a bar chart comparing pickup volumes.
    """
    fig = px.bar(
        counts,
        x="Cluster Group",
        y="Trip Count",
        color="Cluster Group",
        color_discrete_map=color_map,
        title="Trip Volume by Hotspot"
    )
    fig.update_layout(
        xaxis=dict(title="Cluster Group"),
        yaxis=dict(title="Trip Count")
    )
    return fig
