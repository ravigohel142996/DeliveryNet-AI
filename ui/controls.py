"""
ui/controls.py
--------------
Streamlit sidebar controls for the DeliveryNet AI dashboard.

All Streamlit widgets for user input live here, keeping `dashboard.py`
and `app.py` clean of widget boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import DEFAULT_CONFIG


@dataclass
class SimulationControls:
    """Encapsulates all values selected by the user in the sidebar."""

    num_vehicles: int
    num_deliveries: int
    num_delivery_locations: int
    time_steps: int
    time_weight: float
    distance_weight: float
    random_seed: int
    run_simulation: bool


def render_sidebar() -> SimulationControls:
    """
    Render the sidebar and return the chosen control values.

    Returns
    -------
    SimulationControls  dataclass with all current widget values.
    """
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/delivery-truck.png",
        width=80,
    )
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🏙️ City Network")
    num_delivery_locations = st.sidebar.slider(
        "Delivery Locations",
        min_value=5,
        max_value=40,
        value=DEFAULT_CONFIG.city.num_delivery_locations,
        step=1,
        help="Number of delivery destination nodes in the city graph.",
    )

    st.sidebar.subheader("🚚 Fleet")
    num_vehicles = st.sidebar.slider(
        "Fleet Size (Vehicles)",
        min_value=2,
        max_value=20,
        value=DEFAULT_CONFIG.fleet.num_vehicles,
        step=1,
        help="Total number of delivery vehicles in the simulation.",
    )

    st.sidebar.subheader("📦 Orders")
    num_deliveries = st.sidebar.slider(
        "Number of Deliveries",
        min_value=5,
        max_value=100,
        value=DEFAULT_CONFIG.simulation.num_deliveries,
        step=5,
        help="Total delivery orders to simulate.",
    )

    st.sidebar.subheader("⏱️ Simulation")
    time_steps = st.sidebar.slider(
        "Time Steps",
        min_value=10,
        max_value=200,
        value=DEFAULT_CONFIG.simulation.time_steps,
        step=10,
        help="Number of simulation ticks to run.",
    )

    st.sidebar.subheader("🔀 Route Optimizer")
    time_weight = st.sidebar.slider(
        "Time Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Weight given to travel time vs distance in route cost function.",
    )
    distance_weight = round(1.0 - time_weight, 1)
    st.sidebar.caption(f"Distance Weight (auto): **{distance_weight}**")

    st.sidebar.subheader("🎲 Reproducibility")
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=DEFAULT_CONFIG.city.random_seed,
        step=1,
        help="Seed for all random number generators.",
    )

    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "▶ Run Simulation",
        use_container_width=True,
        type="primary",
    )

    return SimulationControls(
        num_vehicles=int(num_vehicles),
        num_deliveries=int(num_deliveries),
        num_delivery_locations=int(num_delivery_locations),
        time_steps=int(time_steps),
        time_weight=float(time_weight),
        distance_weight=float(distance_weight),
        random_seed=int(random_seed),
        run_simulation=run_simulation,
    )
