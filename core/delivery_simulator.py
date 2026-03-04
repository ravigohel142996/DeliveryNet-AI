"""
core/delivery_simulator.py
--------------------------
Orchestrates the end-to-end delivery simulation.

Each call to `step()` advances the simulation clock by one unit and:
  1. Updates traffic conditions in the city network.
  2. Spawns new delivery orders (probabilistically).
  3. Assigns pending orders to fleet vehicles.
  4. Moves vehicles one hop along their assigned route.
  5. Completes deliveries when vehicles arrive at destinations.
  6. Records per-step metrics for dashboard analytics.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config import DEFAULT_CONFIG, FleetConfig, SimulationConfig
from core.city_network import CityNetwork
from core.fleet_manager import DeliveryOrder, FleetManager, VehicleStatus
from core.route_optimizer import RouteOptimizer
from core.traffic_model import TrafficModel


@dataclass
class StepMetrics:
    """Snapshot of simulation KPIs at a single time step."""

    step: int
    hour_of_day: int
    weather: str
    deliveries_completed: int
    deliveries_pending: int
    deliveries_failed: int
    fleet_utilisation: float
    avg_fuel_level: float
    total_distance_km: float


class DeliverySimulator:
    """
    Master simulation engine that integrates all sub-systems.

    Parameters
    ----------
    city_network    : CityNetwork
    traffic_model   : TrafficModel
    fleet_manager   : FleetManager
    sim_cfg         : SimulationConfig
    fleet_cfg       : FleetConfig
    """

    def __init__(
        self,
        city_network: CityNetwork,
        traffic_model: TrafficModel,
        fleet_manager: FleetManager,
        sim_cfg: SimulationConfig = DEFAULT_CONFIG.simulation,
        fleet_cfg: FleetConfig = DEFAULT_CONFIG.fleet,
    ) -> None:
        self._network = city_network
        self._traffic = traffic_model
        self._fleet = fleet_manager
        self._optimizer = RouteOptimizer(city_network)
        self._sim_cfg = sim_cfg
        self._fleet_cfg = fleet_cfg
        self._rng = random.Random(sim_cfg.random_seed)

        self.current_step: int = 0
        self.metrics_history: List[StepMetrics] = []
        self._vehicle_routes: Dict[str, List[str]] = {}   # vehicle_id -> remaining hops
        self._pending_orders: List[DeliveryOrder] = []    # orders not yet released

        # Pre-generate all delivery orders
        self._seed_orders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> StepMetrics:
        """
        Advance simulation by one time step.

        Returns
        -------
        StepMetrics snapshot for this step.
        """
        hour = self._step_to_hour(self.current_step)
        weather = self._weather_for_step(self.current_step)

        # 1. Update traffic conditions
        self._traffic.update(hour, weather)

        # 2. Release orders whose scheduled creation step has been reached
        self._release_due_orders()

        # 3. Assign pending orders to available vehicles and plan routes
        self._fleet.assign_pending_orders()
        self._plan_routes_for_assigned_orders()

        # 3. Move vehicles along routes and complete deliveries
        self._advance_vehicles()

        # 4. Collect metrics
        metrics = self._collect_metrics(hour, weather)
        self.metrics_history.append(metrics)
        self.current_step += 1
        return metrics

    def run_all(self) -> List[StepMetrics]:
        """Run simulation for the full configured number of steps."""
        for _ in range(self._sim_cfg.time_steps):
            self.step()
        return self.metrics_history

    def metrics_dataframe(self) -> pd.DataFrame:
        """Return the accumulated metrics as a tidy DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "step": m.step,
                    "hour_of_day": m.hour_of_day,
                    "weather": m.weather,
                    "deliveries_completed": m.deliveries_completed,
                    "deliveries_pending": m.deliveries_pending,
                    "deliveries_failed": m.deliveries_failed,
                    "fleet_utilisation": m.fleet_utilisation,
                    "avg_fuel_level": m.avg_fuel_level,
                    "total_distance_km": m.total_distance_km,
                }
                for m in self.metrics_history
            ]
        )

    @property
    def delivery_success_rate(self) -> float:
        total = len(self._fleet.orders) + len(self._pending_orders)
        if total == 0:
            return 0.0
        completed = sum(
            1 for o in self._fleet.orders.values() if o.status == "delivered"
        )
        return round(completed / total, 4)

    @property
    def average_delivery_time(self) -> float:
        """Average steps between order creation and completion."""
        completed = [
            o for o in self._fleet.orders.values()
            if o.status == "delivered" and o.completed_at_step is not None
        ]
        if not completed:
            return 0.0
        durations = [o.completed_at_step - o.created_at_step for o in completed]
        return round(sum(durations) / len(durations), 2)

    @property
    def delayed_deliveries(self) -> int:
        """Count orders that took longer than a threshold to complete."""
        threshold = self._sim_cfg.time_steps // 4  # 25% of total steps
        return sum(
            1 for o in self._fleet.orders.values()
            if o.status == "delivered"
            and o.completed_at_step is not None
            and (o.completed_at_step - o.created_at_step) > threshold
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_orders(self) -> None:
        """Pre-generate all delivery orders and store in pending queue."""
        priorities = list(self._sim_cfg.delivery_priority_levels)
        weights = list(self._sim_cfg.priority_weights)
        warehouses = self._network.warehouses
        delivery_nodes = self._network.delivery_nodes

        for i in range(self._sim_cfg.num_deliveries):
            origin = self._rng.choice(warehouses)
            destination = self._rng.choice(delivery_nodes)
            priority = self._rng.choices(priorities, weights=weights, k=1)[0]
            payload = self._rng.randint(1, 5)
            # Stagger order creation across first half of simulation steps
            created_step = self._rng.randint(0, self._sim_cfg.time_steps // 2)
            order = DeliveryOrder(
                order_id=f"ORD-{i:04d}",
                origin=origin,
                destination=destination,
                priority=priority,
                payload_units=payload,
                created_at_step=created_step,
            )
            self._pending_orders.append(order)

    def _release_due_orders(self) -> None:
        """Move orders whose creation step <= current_step into the fleet manager."""
        due = [o for o in self._pending_orders if o.created_at_step <= self.current_step]
        for order in due:
            self._fleet.add_order(order)
            self._pending_orders.remove(order)

    def _plan_routes_for_assigned_orders(self) -> None:
        """Compute routes for vehicles that just received assignments."""
        for vehicle in self._fleet.vehicles.values():
            if vehicle.vehicle_id in self._vehicle_routes:
                continue  # already has a route
            if vehicle.status != VehicleStatus.EN_ROUTE:
                continue
            if not vehicle.deliveries_pending:
                continue

            # Build a multi-stop route: current_location → each destination
            destinations = [
                self._fleet.orders[oid].destination
                for oid in vehicle.deliveries_pending
                if oid in self._fleet.orders
            ]
            stops = [vehicle.current_location] + destinations
            route = self._optimizer.multi_stop_route(stops)
            if route.feasible:
                # Store remaining hops (skip the starting node already at)
                self._vehicle_routes[vehicle.vehicle_id] = route.path[1:]
                vehicle.assigned_route = route.path
            else:
                vehicle.status = VehicleStatus.IDLE

    def _advance_vehicles(self) -> None:
        """Move each active vehicle one hop and handle deliveries on arrival."""
        for vehicle in self._fleet.vehicles.values():
            if vehicle.status not in (VehicleStatus.EN_ROUTE, VehicleStatus.DELIVERING):
                continue
            remaining = self._vehicle_routes.get(vehicle.vehicle_id, [])
            if not remaining:
                vehicle.status = VehicleStatus.IDLE
                del self._vehicle_routes[vehicle.vehicle_id]
                continue

            next_node = remaining[0]
            edge_data = self._network.graph.get_edge_data(
                vehicle.current_location, next_node
            )
            dist_km = edge_data["distance_km"] if edge_data else 0.5

            vehicle.consume_fuel(dist_km, self._fleet_cfg.fuel_consumption_per_km)
            vehicle.current_location = next_node
            self._vehicle_routes[vehicle.vehicle_id] = remaining[1:]

            # Check if any pending delivery has this node as destination
            self._check_deliveries_at(vehicle, next_node)

    def _check_deliveries_at(self, vehicle, node: str) -> None:
        """Complete any deliveries whose destination is the current node."""
        to_complete = [
            oid for oid in list(vehicle.deliveries_pending)
            if self._fleet.orders.get(oid) is not None
            and self._fleet.orders[oid].destination == node
        ]
        for oid in to_complete:
            self._fleet.complete_delivery(oid, self.current_step)

    def _collect_metrics(self, hour: int, weather: str) -> StepMetrics:
        orders = list(self._fleet.orders.values())
        unreleased = len(self._pending_orders)
        total_dist = sum(v.total_distance_km for v in self._fleet.vehicles.values())
        return StepMetrics(
            step=self.current_step,
            hour_of_day=hour,
            weather=weather,
            deliveries_completed=sum(1 for o in orders if o.status == "delivered"),
            deliveries_pending=sum(1 for o in orders if o.status in ("pending", "assigned", "in_transit")) + unreleased,
            deliveries_failed=sum(1 for o in orders if o.status == "failed"),
            fleet_utilisation=self._fleet.fleet_utilisation,
            avg_fuel_level=self._fleet.average_fuel_level,
            total_distance_km=round(total_dist, 2),
        )

    @staticmethod
    def _step_to_hour(step: int) -> int:
        """Map simulation step to hour of day (0-23), cycling every 24 steps."""
        return step % 24

    def _weather_for_step(self, step: int) -> str:
        """
        Deterministically assign weather per step block.
        Changes every ~8 steps for a realistic effect.
        """
        weather_options = list(DEFAULT_CONFIG.traffic.weather_conditions)
        idx = (step // 8) % len(weather_options)
        return weather_options[idx]
