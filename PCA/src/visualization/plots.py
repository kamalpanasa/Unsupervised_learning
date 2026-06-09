import plotly.express as px

def plot_pca_projection_2d(df, color_map, search_hero):
    """
    Renders 2D scatter PCA map.
    """
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Alignment",
        color_discrete_map=color_map,
        hover_data=["Name", "Total", "Intelligence", "Strength", "Speed", "Durability", "Power", "Combat"],
        size="Marker Size" if search_hero else None,
        size_max=20,
        title="2D Projection of Superhero Power Space"
    )
    fig.update_layout(
        xaxis=dict(title="PC1 (Overall Physical Capacity)"),
        yaxis=dict(title="PC2 (Tactical Intelligence vs Raw Strength)")
    )
    return fig

def plot_pca_projection_3d(df, color_map, search_hero):
    """
    Renders 3D scatter PCA map.
    """
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Alignment",
        color_discrete_map=color_map,
        hover_data=["Name", "Total", "Intelligence", "Strength", "Speed", "Durability", "Power", "Combat"],
        size="Marker Size" if search_hero else None,
        size_max=20,
        title="3D Projection of Superhero Power Space"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_loadings_bar(loadings_melted):
    """
    Plots attribute loadings per Principal Component.
    """
    fig = px.bar(
        loadings_melted,
        x="Attribute",
        y="Weight",
        color="Component",
        barmode="group",
        title="Weight Analysis of Base Stats across PCs"
    )
    return fig

def plot_scree_plot(var_df):
    """
    Plots Scree cumulative variance curve.
    """
    fig = px.line(
        var_df,
        x="Component",
        y="Cumulative Variance",
        markers=True,
        title="Cumulative Explained Variance Curve (Scree Plot)"
    )
    fig.update_yaxes(range=[0, 1.1])
    return fig
