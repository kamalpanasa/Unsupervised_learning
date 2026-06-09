import plotly.express as px

def plot_tsne_2d(df, style_colors):
    """
    Renders 2D t-SNE scatter plot.
    """
    fig = px.scatter(
        df,
        x="tSNE_1",
        y="tSNE_2",
        color="Style_Group",
        color_discrete_map=style_colors,
        hover_data=["Character_ID", "Hair_Color", "Eye_Color", "Jaw_Width", "Eye_Size", "Smile_Score", "Blush_Intensity"],
        title="2D t-SNE Embedding of Anime Faces"
    )
    fig.update_layout(
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2")
    )
    return fig

def plot_tsne_3d(df, style_colors):
    """
    Renders 3D t-SNE scatter plot.
    """
    fig = px.scatter_3d(
        df,
        x="tSNE_1",
        y="tSNE_2",
        z="tSNE_3",
        color="Style_Group",
        color_discrete_map=style_colors,
        hover_data=["Character_ID", "Hair_Color", "Eye_Color", "Jaw_Width", "Eye_Size", "Smile_Score", "Blush_Intensity"],
        title="3D t-SNE Embedding of Anime Faces"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig
