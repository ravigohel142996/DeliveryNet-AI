"""
ui/dashboard.py
---------------
Streamlit page layout and rendering logic for the DeliveryNet AI dashboard.

This module contains only UI composition code.  All business logic lives in
the core/ and models/ layers, keeping this file clean and easy to iterate.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from config import DEFAULT_CONFIG, AppConfig, CityConfig, FleetConfig, SimulationConfig
from core.city_network import CityNetwork
from core.delivery_simulator import DeliverySimulator
from core.fleet_manager import FleetManager
from core.traffic_model import TrafficModel
from core.route_optimizer import RouteOptimizer
from ui.charts import (
    build_delivery_timeline,
    build_feature_importance_chart,
    build_fleet_utilisation_chart,
    build_network_figure,
    build_order_status_pie,
    build_traffic_heatmap,
)
from ui.controls import SimulationControls
from utils.helpers import format_distance, format_duration, pct, summarise_fleet, summarise_orders


# ---------------------------------------------------------------------------
# Page-level configuration (must be called once from app.py)
# ---------------------------------------------------------------------------

def configure_page() -> None:
    """Set Streamlit page metadata (must be the very first Streamlit call)."""
    st.set_page_config(
        page_title=DEFAULT_CONFIG.dashboard.page_title,
        page_icon=DEFAULT_CONFIG.dashboard.page_icon,
        layout=DEFAULT_CONFIG.dashboard.layout,
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "simulator": None,
        "network": None,
        "traffic_model": None,
        "fleet_manager": None,
        "sim_ran": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Simulation bootstrap
# ---------------------------------------------------------------------------

def _build_simulation(controls: SimulationControls) -> DeliverySimulator:
    """Instantiate all simulation components from user controls."""
    city_cfg = CityConfig(
        num_delivery_locations=controls.num_delivery_locations,
        random_seed=controls.random_seed,
    )
    fleet_cfg = FleetConfig(num_vehicles=controls.num_vehicles)
    sim_cfg = SimulationConfig(
        num_deliveries=controls.num_deliveries,
        time_steps=controls.time_steps,
        random_seed=controls.random_seed,
    )

    network = CityNetwork(city_cfg)
    traffic = TrafficModel(network)
    fleet = FleetManager(
        warehouses=network.warehouses,
        fleet_cfg=fleet_cfg,
        sim_cfg=sim_cfg,
    )
    simulator = DeliverySimulator(
        city_network=network,
        traffic_model=traffic,
        fleet_manager=fleet,
        sim_cfg=sim_cfg,
        fleet_cfg=fleet_cfg,
    )
    return simulator


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_dashboard(controls: SimulationControls) -> None:
    """
    Render the full dashboard.  Called from app.py on every Streamlit rerun.

    Parameters
    ----------
    controls : SimulationControls  current sidebar values
    """
    _init_session_state()

    # Header
    st.markdown(
        "## 🚚 DeliveryNet AI — Intelligent Logistics Route Optimisation",
        unsafe_allow_html=False,
    )
    st.caption(
        "AI-powered city delivery simulation · Traffic prediction · Route optimisation · Fleet management"
    )
    st.markdown("---")

    # Run simulation when button pressed
    if controls.run_simulation:
        with st.spinner("⚙️ Building city network and training traffic model…"):
            simulator = _build_simulation(controls)
        with st.spinner("▶ Running simulation…"):
            simulator.run_all()
        st.session_state["simulator"] = simulator
        st.session_state["sim_ran"] = True
        st.success("✅ Simulation complete!", icon="✅")

    simulator: Optional[DeliverySimulator] = st.session_state.get("simulator")

    if simulator is None:
        _render_welcome()
        return

    _render_kpis(simulator)
    st.markdown("---")
    _render_network_and_traffic(simulator)
    st.markdown("---")
    _render_timeline_charts(simulator)
    st.markdown("---")
    _render_fleet_and_orders(simulator)
    st.markdown("---")
    _render_ml_insights(simulator)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_welcome() -> None:
    st.info(
        "👈  Configure the simulation parameters in the sidebar and press **▶ Run Simulation** to begin.",
        icon="ℹ️",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏙️ City Network\nRandomly generated graph with warehouses and delivery locations connected by roads with realistic distance and travel-time weights.")
    with col2:
        st.markdown("### 🤖 Traffic AI\nRandom Forest Regressor trained on 5 000 synthetic samples predicts congestion based on time-of-day, weather, road type, and traffic density.")
    with col3:
        st.markdown("### 🗺️ Route Optimisation\nDijkstra and A* algorithms minimise a composite cost of travel time and fuel consumption, returning the optimal delivery route.")


def _render_kpis(simulator: DeliverySimulator) -> None:
    """Top-row KPI metric cards."""
    fleet = simulator._fleet
    total_orders = len(fleet.orders)
    completed = sum(1 for o in fleet.orders.values() if o.status == "delivered")
    pending = sum(1 for o in fleet.orders.values() if o.status in ("pending", "assigned"))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📦 Total Orders", total_orders)
    col2.metric("✅ Delivered", completed)
    col3.metric("⏳ Pending", pending)
    col4.metric(
        "📊 Success Rate",
        pct(simulator.delivery_success_rate),
    )
    col5.metric(
        "⏱️ Avg Delivery Time",
        f"{simulator.average_delivery_time:.1f} steps",
    )

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("🚚 Fleet Size", len(fleet.vehicles))
    col7.metric("⚡ Fleet Utilisation", pct(fleet.fleet_utilisation))
    col8.metric("⚠️ Delayed", simulator.delayed_deliveries)
    col9.metric("⛽ Avg Fuel", f"{fleet.average_fuel_level:.1f} L")
    col10.metric(
        "🛣️ Total Distance",
        format_distance(sum(v.total_distance_km for v in fleet.vehicles.values())),
    )


def _render_network_and_traffic(simulator: DeliverySimulator) -> None:
    st.subheader("🗺️ City Network & Traffic")
    col_map, col_heat = st.columns([3, 2])

    with col_map:
        network_fig = build_network_figure(simulator._network.graph)
        st.plotly_chart(network_fig, use_container_width=True)

    with col_heat:
        traffic_df = simulator._traffic.get_edge_traffic_dataframe()
        heat_fig = build_traffic_heatmap(traffic_df)
        st.plotly_chart(heat_fig, use_container_width=True)

        # Traffic stats
        if not traffic_df.empty:
            avg_cong = traffic_df["congestion"].mean()
            st.caption(
                f"**Avg Congestion:** {avg_cong:.2f}x  |  "
                f"**Hour:** {simulator._traffic.current_hour:02d}:00  |  "
                f"**Weather:** {simulator._traffic.current_weather.title()}"
            )


def _render_timeline_charts(simulator: DeliverySimulator) -> None:
    st.subheader("📈 Simulation Analytics")
    metrics_df = simulator.metrics_dataframe()
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(build_delivery_timeline(metrics_df), use_container_width=True)
    with col2:
        st.plotly_chart(build_fleet_utilisation_chart(metrics_df), use_container_width=True)


def _render_fleet_and_orders(simulator: DeliverySimulator) -> None:
    st.subheader("🚚 Fleet & Orders")
    col_fleet, col_orders, col_pie = st.columns([2, 2, 1])

    with col_fleet:
        st.markdown("**Fleet Status**")
        fleet_df = summarise_fleet(simulator._fleet.vehicles)
        st.dataframe(fleet_df, use_container_width=True, hide_index=True)

    with col_orders:
        st.markdown("**Order Summary**")
        order_df = summarise_orders(simulator._fleet.orders)
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    with col_pie:
        pie = build_order_status_pie(simulator._fleet.orders)
        st.plotly_chart(pie, use_container_width=True)


def _render_ml_insights(simulator: DeliverySimulator) -> None:
    st.subheader("🤖 Traffic Prediction Model Insights")
    metrics = simulator._traffic.model_metrics
    importances = simulator._traffic.feature_importances

    col_m, col_f = st.columns([1, 2])
    with col_m:
        st.markdown("**Model Performance**")
        st.metric("Train MAE", f"{metrics['train_mae']:.4f}")
        st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
        st.caption(
            "Lower MAE and higher R² indicate better congestion prediction accuracy."
        )
    with col_f:
        fig = build_feature_importance_chart(importances)
        st.plotly_chart(fig, use_container_width=True)
