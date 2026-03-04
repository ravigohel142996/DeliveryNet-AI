"""
ui/charts.py
------------
All Plotly figure factories for the DeliveryNet AI dashboard.

Each function is pure: it accepts data arguments and returns a
`plotly.graph_objects.Figure` with no side effects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DEFAULT_CONFIG, DashboardConfig


_DASH_CFG: DashboardConfig = DEFAULT_CONFIG.dashboard


def build_network_figure(
    graph: nx.DiGraph,
    highlight_path: Optional[List[str]] = None,
    title: str = "City Logistics Network",
) -> go.Figure:
    """
    Render the city graph as a geo-scatter map.

    Parameters
    ----------
    graph          : nx.DiGraph    the city network graph
    highlight_path : list[str]     optional path to draw in a contrasting colour
    title          : str
    """
    fig = go.Figure()

    # --- Draw all edges ---
    for u, v, data in graph.edges(data=True):
        lat1, lon1 = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat2, lon2 = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        congestion = data.get("congestion", 1.0)
        colour = _congestion_colour(congestion)
        fig.add_trace(
            go.Scattergeo(
                lon=[lon1, lon2, None],
                lat=[lat1, lat2, None],
                mode="lines",
                line=dict(width=1.5, color=colour),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # --- Highlight route ---
    if highlight_path and len(highlight_path) >= 2:
        path_lons = [graph.nodes[n]["lon"] for n in highlight_path]
        path_lats = [graph.nodes[n]["lat"] for n in highlight_path]
        fig.add_trace(
            go.Scattergeo(
                lon=path_lons,
                lat=path_lats,
                mode="lines+markers",
                line=dict(width=4, color=_DASH_CFG.route_colour),
                marker=dict(size=8, color=_DASH_CFG.route_colour),
                name="Optimal Route",
            )
        )

    # --- Draw nodes ---
    warehouse_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "warehouse"]
    delivery_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "delivery"]

    for nodes, colour, symbol, layer_name in [
        (delivery_nodes, _DASH_CFG.delivery_node_colour, "circle", "Delivery Loc"),
        (warehouse_nodes, _DASH_CFG.warehouse_colour, "square", "Warehouse"),
    ]:
        fig.add_trace(
            go.Scattergeo(
                lon=[graph.nodes[n]["lon"] for n in nodes],
                lat=[graph.nodes[n]["lat"] for n in nodes],
                mode="markers+text",
                marker=dict(size=12, color=colour, symbol=symbol),
                text=[graph.nodes[n]["label"] for n in nodes],
                textposition="top center",
                name=layer_name,
                hovertext=[
                    f"{graph.nodes[n]['label']}<br>({graph.nodes[n]['lat']:.4f}, {graph.nodes[n]['lon']:.4f})"
                    for n in nodes
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        geo=dict(
            scope="north america",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcoastlines=True,
            coastlinecolor="rgb(180,180,180)",
            projection_type="mercator",
            lonaxis=dict(range=[-74.05, -73.89]),
            lataxis=dict(range=[40.68, 40.82]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02),
        height=500,
    )
    return fig


def build_traffic_heatmap(
    traffic_df: pd.DataFrame,
    title: str = "Traffic Congestion Heatmap",
) -> go.Figure:
    """
    Render a heatmap of congestion levels by road type and hour bucket.

    Parameters
    ----------
    traffic_df : pd.DataFrame  output of TrafficModel.get_edge_traffic_dataframe()
    """
    if traffic_df.empty:
        return _empty_figure(title)

    pivot = (
        traffic_df
        .groupby("road_type")["congestion"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={"mean": "Avg Congestion", "max": "Max Congestion", "min": "Min Congestion"})
    )

    fig = px.bar(
        pivot,
        x="road_type",
        y="Avg Congestion",
        color="Avg Congestion",
        color_continuous_scale=_DASH_CFG.heatmap_colorscale,
        range_color=[1.0, 3.5],
        title=title,
        labels={"road_type": "Road Type", "Avg Congestion": "Avg Congestion Factor"},
        error_y=pivot["Max Congestion"] - pivot["Avg Congestion"],
        error_y_minus=pivot["Avg Congestion"] - pivot["Min Congestion"],
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def build_delivery_timeline(
    metrics_df: pd.DataFrame,
    title: str = "Delivery Completion Over Time",
) -> go.Figure:
    """
    Line chart of cumulative deliveries completed versus simulation step.

    Parameters
    ----------
    metrics_df : pd.DataFrame  output of DeliverySimulator.metrics_dataframe()
    """
    if metrics_df.empty:
        return _empty_figure(title)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics_df["step"],
            y=metrics_df["deliveries_completed"],
            mode="lines+markers",
            name="Completed",
            line=dict(color=_DASH_CFG.route_colour, width=2),
            fill="tozeroy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df["step"],
            y=metrics_df["deliveries_pending"],
            mode="lines",
            name="Pending",
            line=dict(color=_DASH_CFG.vehicle_colour, width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Simulation Step",
        yaxis_title="Number of Deliveries",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def build_fleet_utilisation_chart(
    metrics_df: pd.DataFrame,
    title: str = "Fleet Utilisation & Fuel Level",
) -> go.Figure:
    """
    Dual-axis chart: fleet utilisation (%) and average fuel level over time.

    Parameters
    ----------
    metrics_df : pd.DataFrame  output of DeliverySimulator.metrics_dataframe()
    """
    if metrics_df.empty:
        return _empty_figure(title)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metrics_df["step"],
            y=metrics_df["fleet_utilisation"] * 100,
            name="Fleet Utilisation (%)",
            mode="lines",
            line=dict(color=_DASH_CFG.warehouse_colour, width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df["step"],
            y=metrics_df["avg_fuel_level"],
            name="Avg Fuel Level (L)",
            mode="lines",
            line=dict(color=_DASH_CFG.delivery_node_colour, width=2, dash="dash"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Simulation Step",
        yaxis=dict(title="Utilisation (%)", range=[0, 110]),
        yaxis2=dict(title="Fuel (L)", overlaying="y", side="right", range=[0, 105]),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def build_feature_importance_chart(
    importances: pd.Series,
    title: str = "Traffic Model – Feature Importances",
) -> go.Figure:
    """
    Horizontal bar chart of Random Forest feature importances.

    Parameters
    ----------
    importances : pd.Series  index = feature names, values = importance scores
    """
    fig = px.bar(
        importances.rename("importance").reset_index().rename(columns={"index": "feature"}),
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"feature": "Feature", "importance": "Importance"},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def build_order_status_pie(
    orders: Dict,
    title: str = "Order Status Distribution",
) -> go.Figure:
    """
    Pie chart of delivery order statuses.

    Parameters
    ----------
    orders : dict {order_id: DeliveryOrder}
    """
    if not orders:
        return _empty_figure(title)

    status_counts: Dict[str, int] = {}
    for o in orders.values():
        status_counts[o.status] = status_counts.get(o.status, 0) + 1

    fig = go.Figure(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker=dict(
                colors=[
                    "#00CC96" if s == "delivered" else
                    "#FFA15A" if s == "assigned" else
                    "#636EFA" if s == "pending" else
                    "#EF553B"
                    for s in status_counts.keys()
                ]
            ),
        )
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _congestion_colour(factor: float) -> str:
    """Map 1.0–3.5 congestion factor to a hex colour (green→red)."""
    norm = min(max((factor - 1.0) / 2.5, 0.0), 1.0)
    r = int(255 * norm)
    g = int(255 * (1.0 - norm))
    return f"rgb({r},{g},0)"


def _empty_figure(title: str) -> go.Figure:
    """Return an empty placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="No data available yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="grey"),
            )
        ],
        height=300,
    )
    return fig
