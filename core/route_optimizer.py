"""
core/route_optimizer.py
-----------------------
Provides Dijkstra and A* route optimisation over the city graph.

Both algorithms minimise a composite cost that considers travel time
and an optional fuel-efficiency penalty for long-distance segments.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import networkx as nx

from core.city_network import CityNetwork


class RouteResult(NamedTuple):
    """Immutable result object returned by both optimisation algorithms."""

    path: List[str]
    total_time_min: float
    total_distance_km: float
    algorithm: str
    feasible: bool


class RouteOptimizer:
    """
    Wraps Dijkstra and A* shortest-path algorithms with a unified interface.

    The cost function can weight travel time and distance independently to
    balance speed versus fuel efficiency.

    Parameters
    ----------
    network         : CityNetwork
    time_weight     : float   weight applied to travel_time in composite cost
    distance_weight : float   weight applied to distance_km in composite cost
    """

    def __init__(
        self,
        network: CityNetwork,
        time_weight: float = 0.7,
        distance_weight: float = 0.3,
    ) -> None:
        if not math.isclose(time_weight + distance_weight, 1.0, abs_tol=1e-6):
            raise ValueError("time_weight + distance_weight must equal 1.0")
        self._network = network
        self._time_w = time_weight
        self._dist_w = distance_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dijkstra(self, source: str, target: str) -> RouteResult:
        """
        Run Dijkstra's algorithm using the composite edge cost.

        Returns
        -------
        RouteResult with the optimal path and cost breakdown.
        """
        try:
            path = nx.dijkstra_path(
                self._network.graph, source, target, weight=self._composite_cost_key
            )
            return self._build_result(path, "dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return RouteResult(
                path=[], total_time_min=float("inf"),
                total_distance_km=float("inf"),
                algorithm="dijkstra", feasible=False,
            )

    def astar(self, source: str, target: str) -> RouteResult:
        """
        Run A* algorithm using geographic (Haversine) heuristic.

        The heuristic estimates remaining travel time assuming free-flow
        speed on arterial roads.
        """
        try:
            path = nx.astar_path(
                self._network.graph,
                source,
                target,
                heuristic=self._heuristic,
                weight=self._composite_cost_key,
            )
            return self._build_result(path, "astar")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return RouteResult(
                path=[], total_time_min=float("inf"),
                total_distance_km=float("inf"),
                algorithm="astar", feasible=False,
            )

    def best_route(self, source: str, target: str) -> RouteResult:
        """
        Run both algorithms and return whichever yields lower composite cost.
        """
        d = self.dijkstra(source, target)
        a = self.astar(source, target)

        if not d.feasible and not a.feasible:
            return d  # both infeasible, return either

        candidates = [r for r in (d, a) if r.feasible]
        return min(
            candidates,
            key=lambda r: self._composite_scalar(r.total_time_min, r.total_distance_km),
        )

    def multi_stop_route(self, stops: List[str]) -> RouteResult:
        """
        Build a route that visits all stops in order by chaining best_route
        calls between consecutive stops.

        Note: This is a nearest-neighbour heuristic, not TSP-optimal.
        """
        if len(stops) < 2:
            raise ValueError("At least two stops are required.")

        full_path: List[str] = []
        total_time = 0.0
        total_dist = 0.0
        feasible = True

        for i in range(len(stops) - 1):
            segment = self.best_route(stops[i], stops[i + 1])
            if not segment.feasible:
                feasible = False
                break
            # Avoid duplicating junction nodes
            if full_path:
                full_path.extend(segment.path[1:])
            else:
                full_path.extend(segment.path)
            total_time += segment.total_time_min
            total_dist += segment.total_distance_km

        return RouteResult(
            path=full_path,
            total_time_min=round(total_time, 3),
            total_distance_km=round(total_dist, 4),
            algorithm="multi_stop",
            feasible=feasible,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _composite_cost_key(self, u: str, v: str, edge_data: Dict) -> float:
        """Edge weight function compatible with NetworkX."""
        return (
            self._time_w * edge_data.get("travel_time", 0.0)
            + self._dist_w * edge_data.get("distance_km", 0.0)
        )

    def _composite_scalar(self, time_min: float, dist_km: float) -> float:
        return self._time_w * time_min + self._dist_w * dist_km

    def _heuristic(self, u: str, v: str) -> float:
        """
        A* admissible heuristic: Haversine distance converted to minutes
        assuming 60 km/h free-flow speed.
        """
        g = self._network.graph
        lat1, lon1 = math.radians(g.nodes[u]["lat"]), math.radians(g.nodes[u]["lon"])
        lat2, lon2 = math.radians(g.nodes[v]["lat"]), math.radians(g.nodes[v]["lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        dist_km = 6_371.0 * 2 * math.asin(math.sqrt(a))
        # time at 60 km/h free-flow (minutes) as lower-bound estimate
        return (dist_km / 60.0) * 60.0

    def _build_result(self, path: List[str], algorithm: str) -> RouteResult:
        g = self._network.graph
        total_time = sum(g[u][v]["travel_time"] for u, v in zip(path[:-1], path[1:]))
        total_dist = sum(g[u][v]["distance_km"] for u, v in zip(path[:-1], path[1:]))
        return RouteResult(
            path=path,
            total_time_min=round(total_time, 3),
            total_distance_km=round(total_dist, 4),
            algorithm=algorithm,
            feasible=True,
        )
