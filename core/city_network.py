"""
core/city_network.py
--------------------
Generates a weighted directed graph that models a city logistics network.

Nodes:
    - Warehouses  (node_type = "warehouse")
    - Delivery locations (node_type = "delivery")

Edge attributes:
    - distance_km   : Euclidean approximation in km
    - road_type     : one of (highway | arterial | local | residential)
    - base_time_min : unimpeded travel time in minutes
    - congestion    : current multiplicative congestion factor (1.0 = free flow)
    - travel_time   : effective travel time after applying congestion
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import CityConfig, DEFAULT_CONFIG


class CityNetwork:
    """Builds and exposes a weighted directed graph of a city logistics network."""

    _ROAD_TYPE_WEIGHTS: Dict[str, float] = {
        "highway": 0.15,
        "arterial": 0.30,
        "local": 0.35,
        "residential": 0.20,
    }

    def __init__(self, cfg: CityConfig = DEFAULT_CONFIG.city) -> None:
        self._cfg = cfg
        self._rng = random.Random(cfg.random_seed)
        self._np_rng = np.random.default_rng(cfg.random_seed)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def warehouses(self) -> List[str]:
        """Return node IDs whose type is 'warehouse'."""
        return [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "warehouse"
        ]

    @property
    def delivery_nodes(self) -> List[str]:
        """Return node IDs whose type is 'delivery'."""
        return [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "delivery"
        ]

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Return {node_id: (lon, lat)} suitable for Plotly geo charts."""
        return {
            n: (d["lon"], d["lat"])
            for n, d in self.graph.nodes(data=True)
        }

    def update_congestion(self, hour_of_day: int, weather: str) -> None:
        """
        Recompute edge congestion and travel_time for all edges given current
        simulation conditions.

        Parameters
        ----------
        hour_of_day : int   0-23
        weather     : str   one of the weather condition labels in TrafficConfig
        """
        for u, v, data in self.graph.edges(data=True):
            congestion = self._compute_congestion(hour_of_day, weather, data["road_type"])
            data["congestion"] = congestion
            data["travel_time"] = round(data["base_time_min"] * congestion, 3)

    def shortest_path(
        self,
        source: str,
        target: str,
        weight: str = "travel_time",
    ) -> Optional[List[str]]:
        """Return the shortest path between two nodes or None if unreachable."""
        try:
            return nx.dijkstra_path(self.graph, source, target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def path_cost(
        self,
        path: List[str],
        weight: str = "travel_time",
    ) -> float:
        """Sum the given edge weight along *path*."""
        return sum(
            self.graph[u][v][weight] for u, v in zip(path[:-1], path[1:])
        )

    # ------------------------------------------------------------------
    # Build internals
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self._place_nodes()
        self._connect_edges()
        self._ensure_connectivity()

    def _place_nodes(self) -> None:
        lat_min, lat_max, lon_min, lon_max = self._cfg.geo_bounds

        for i in range(self._cfg.num_warehouses):
            node_id = f"W{i}"
            lat = self._np_rng.uniform(lat_min, lat_max)
            lon = self._np_rng.uniform(lon_min, lon_max)
            self.graph.add_node(
                node_id,
                node_type="warehouse",
                label=f"Warehouse {i}",
                lat=round(lat, 6),
                lon=round(lon, 6),
            )

        for i in range(self._cfg.num_delivery_locations):
            node_id = f"D{i}"
            lat = self._np_rng.uniform(lat_min, lat_max)
            lon = self._np_rng.uniform(lon_min, lon_max)
            self.graph.add_node(
                node_id,
                node_type="delivery",
                label=f"Loc {i}",
                lat=round(lat, 6),
                lon=round(lon, 6),
            )

    def _connect_edges(self) -> None:
        nodes = list(self.graph.nodes())
        road_types = list(self._ROAD_TYPE_WEIGHTS.keys())
        road_weights = list(self._ROAD_TYPE_WEIGHTS.values())

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i >= j:
                    continue
                if self._rng.random() > self._cfg.edge_probability:
                    continue
                road_type = self._rng.choices(road_types, weights=road_weights, k=1)[0]
                dist_km = self._haversine_km(u, v)
                base_speed = self._road_speed_kmh(road_type)
                base_time = round((dist_km / base_speed) * 60.0, 3)  # minutes
                edge_attrs = {
                    "distance_km": round(dist_km, 4),
                    "road_type": road_type,
                    "base_time_min": base_time,
                    "congestion": 1.0,
                    "travel_time": base_time,
                }
                self.graph.add_edge(u, v, **edge_attrs)
                self.graph.add_edge(v, u, **edge_attrs)  # bidirectional

    def _ensure_connectivity(self) -> None:
        """
        Guarantee that the graph is weakly connected by stitching isolated
        components to the largest one with minimal synthetic edges.
        """
        components = list(nx.weakly_connected_components(self.graph))
        if len(components) <= 1:
            return

        largest = max(components, key=len)
        anchor = next(iter(largest))
        for component in components:
            if component is largest:
                continue
            orphan = next(iter(component))
            road_type = "local"
            dist_km = self._haversine_km(anchor, orphan)
            base_time = round((dist_km / self._road_speed_kmh(road_type)) * 60.0, 3)
            attrs = {
                "distance_km": round(dist_km, 4),
                "road_type": road_type,
                "base_time_min": base_time,
                "congestion": 1.0,
                "travel_time": base_time,
            }
            self.graph.add_edge(anchor, orphan, **attrs)
            self.graph.add_edge(orphan, anchor, **attrs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _haversine_km(self, u: str, v: str) -> float:
        lat1 = np.radians(self.graph.nodes[u]["lat"])
        lon1 = np.radians(self.graph.nodes[u]["lon"])
        lat2 = np.radians(self.graph.nodes[v]["lat"])
        lon2 = np.radians(self.graph.nodes[v]["lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6_371.0 * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _road_speed_kmh(road_type: str) -> float:
        speeds = {"highway": 90.0, "arterial": 60.0, "local": 40.0, "residential": 25.0}
        return speeds.get(road_type, 40.0)

    @staticmethod
    def _compute_congestion(hour: int, weather: str, road_type: str) -> float:
        """
        Heuristic congestion multiplier.
        Peak hours and bad weather inflate travel time.
        """
        base = 1.0
        # Time-of-day component
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            base += 0.8
        elif 11 <= hour <= 13:
            base += 0.3
        elif 22 <= hour or hour <= 5:
            base -= 0.2

        # Weather component
        weather_impact = {"clear": 0.0, "rain": 0.3, "fog": 0.5, "snow": 1.0}
        base += weather_impact.get(weather, 0.0)

        # Road-type sensitivity
        sensitivity = {"highway": 0.6, "arterial": 0.8, "local": 1.0, "residential": 1.1}
        factor = 1.0 + (base - 1.0) * sensitivity.get(road_type, 1.0)
        return round(max(1.0, min(factor, 3.5)), 3)
