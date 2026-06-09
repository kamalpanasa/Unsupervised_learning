import plotly.express as px

def plot_anomaly_score_trend(unit_df):
    """
    Plots the unsupervised decision score trend across cycles.
    """
    fig = px.line(
        unit_df,
        x="cycle",
        y="anomaly_score",
        title="Engine Degradation Trend (Anomaly Score over Cycles)",
        labels={"cycle": "Operational Cycles", "anomaly_score": "Anomaly Score"}
    )
    # Add colored anomaly markers
    anom_df = unit_df[unit_df["anomaly"] == 1]
    fig.add_scatter(
        x=anom_df["cycle"],
        y=anom_df["anomaly_score"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Flagged Anomaly"
    )
    return fig

def plot_sensor_telemetry(unit_df, sensor_name):
    """
    Plots raw sensor values with highlighted anomaly cycles.
    """
    fig = px.line(
        unit_df,
        x="cycle",
        y=sensor_name,
        title=f"Telemetry for {sensor_name} (Red markers indicate flagged anomalies)",
        labels={"cycle": "Operational Cycles", sensor_name: "Sensor Reading"}
    )
    anom_df = unit_df[unit_df["anomaly"] == 1]
    fig.add_scatter(
        x=anom_df["cycle"],
        y=anom_df[sensor_name],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Flagged Anomaly"
    )
    return fig

def plot_multivariate_space_3d(unit_df, plot_sensors):
    """
    Projects normal vs anomalous points in 3D sensor space.
    """
    fig = px.scatter_3d(
        unit_df,
        x=plot_sensors[0],
        y=plot_sensors[1],
        z=plot_sensors[2],
        color="Status",
        color_discrete_map={"Normal": "#1F77B4", "Degraded/Anomaly": "#D62728"},
        hover_data=["cycle"],
        title=f"3D Space: {plot_sensors[0]} vs {plot_sensors[1]} vs {plot_sensors[2]}",
        opacity=0.8
      )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig
