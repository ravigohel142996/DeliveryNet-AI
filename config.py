"""
config.py
---------
Central configuration for DeliveryNet AI.
All tunable parameters and constants live here so that no magic numbers
appear anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# City / Graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CityConfig:
    num_warehouses: int = 3
    num_delivery_locations: int = 20
    grid_rows: int = 8
    grid_cols: int = 8
    # probability that any two adjacent nodes share a road edge
    edge_probability: float = 0.65
    # geographic bounding box (lat_min, lat_max, lon_min, lon_max)
    geo_bounds: Tuple[float, float, float, float] = (40.70, 40.80, -74.02, -73.92)
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrafficConfig:
    # synthetic training dataset size
    training_samples: int = 5_000
    # road type labels
    road_types: Tuple[str, ...] = ("highway", "arterial", "local", "residential")
    # weather condition labels
    weather_conditions: Tuple[str, ...] = ("clear", "rain", "fog", "snow")
    # peak-hour windows (start_hour, end_hour)
    morning_peak: Tuple[int, int] = (7, 9)
    evening_peak: Tuple[int, int] = (16, 19)
    # multiplicative congestion range applied to base travel time
    congestion_min: float = 1.0
    congestion_max: float = 3.5
    random_seed: int = 7


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FleetConfig:
    num_vehicles: int = 8
    vehicle_capacity_range: Tuple[int, int] = (10, 30)   # units
    fuel_tank_size: float = 100.0                         # litres
    fuel_consumption_per_km: float = 0.12                # litres / km
    max_deliveries_per_vehicle: int = 10
    vehicle_speed_kmh: float = 40.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    num_deliveries: int = 30
    time_steps: int = 50
    delivery_priority_levels: Tuple[str, ...] = ("low", "medium", "high", "urgent")
    # probability weights for priority assignment
    priority_weights: Tuple[float, ...] = (0.30, 0.40, 0.20, 0.10)
    random_seed: int = 99


# ---------------------------------------------------------------------------
# UI / Dashboard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardConfig:
    page_title: str = "DeliveryNet AI"
    page_icon: str = "🚚"
    layout: str = "wide"
    # Plotly colour palette
    warehouse_colour: str = "#EF553B"
    delivery_node_colour: str = "#636EFA"
    route_colour: str = "#00CC96"
    vehicle_colour: str = "#FFA15A"
    heatmap_colorscale: str = "RdYlGn_r"


# ---------------------------------------------------------------------------
# Aggregate configuration singleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    city: CityConfig = field(default_factory=CityConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# Module-level default instance – import this everywhere
DEFAULT_CONFIG = AppConfig()
