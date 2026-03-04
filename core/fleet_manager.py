"""
core/fleet_manager.py
---------------------
Models a fleet of delivery vehicles and assigns deliveries to them
based on proximity, capacity, and delivery priority.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from config import DEFAULT_CONFIG, FleetConfig, SimulationConfig


class VehicleStatus(Enum):
    IDLE = auto()
    EN_ROUTE = auto()
    DELIVERING = auto()
    RETURNING = auto()
    OUT_OF_FUEL = auto()


@dataclass
class Vehicle:
    """Represents a single fleet vehicle with its operational state."""

    vehicle_id: str
    capacity: int
    fuel_level: float
    current_location: str
    home_base: str
    status: VehicleStatus = VehicleStatus.IDLE
    assigned_route: List[str] = field(default_factory=list)
    deliveries_pending: List[str] = field(default_factory=list)
    deliveries_completed: int = 0
    total_distance_km: float = 0.0
    total_fuel_consumed: float = 0.0

    @property
    def available_capacity(self) -> int:
        return self.capacity - len(self.deliveries_pending)

    @property
    def fuel_percentage(self) -> float:
        return round(self.fuel_level / 100.0 * 100, 1)

    @property
    def is_available(self) -> bool:
        return self.status in (VehicleStatus.IDLE, VehicleStatus.EN_ROUTE) and self.available_capacity > 0

    def consume_fuel(self, distance_km: float, consumption_per_km: float) -> None:
        consumed = distance_km * consumption_per_km
        self.fuel_level = max(0.0, self.fuel_level - consumed)
        self.total_fuel_consumed += consumed
        self.total_distance_km += distance_km
        if self.fuel_level <= 0.0:
            self.status = VehicleStatus.OUT_OF_FUEL


@dataclass
class DeliveryOrder:
    """Represents a single delivery task."""

    order_id: str
    origin: str           # warehouse node
    destination: str      # delivery location node
    priority: str         # low | medium | high | urgent
    payload_units: int    # size in capacity units
    assigned_vehicle: Optional[str] = None
    status: str = "pending"  # pending | assigned | in_transit | delivered | failed
    created_at_step: int = 0
    completed_at_step: Optional[int] = None

    @property
    def priority_score(self) -> int:
        scores = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
        return scores.get(self.priority, 1)


class FleetManager:
    """
    Manages the fleet of vehicles and the pool of delivery orders.

    Responsibilities
    ----------------
    - Spawn vehicles at simulation start.
    - Accept new delivery orders.
    - Assign orders to the most suitable available vehicle.
    - Track fleet-wide KPI metrics.
    """

    def __init__(
        self,
        warehouses: List[str],
        fleet_cfg: FleetConfig = DEFAULT_CONFIG.fleet,
        sim_cfg: SimulationConfig = DEFAULT_CONFIG.simulation,
    ) -> None:
        self._fleet_cfg = fleet_cfg
        self._sim_cfg = sim_cfg
        self._warehouses = warehouses
        self._rng = random.Random(fleet_cfg.num_vehicles)

        self.vehicles: Dict[str, Vehicle] = {}
        self.orders: Dict[str, DeliveryOrder] = {}
        self._initialise_fleet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_order(self, order: DeliveryOrder) -> None:
        """Register a new delivery order."""
        self.orders[order.order_id] = order

    def assign_pending_orders(self) -> int:
        """
        Assign all unassigned orders to the best available vehicle.

        Assignment heuristic
        --------------------
        1. Sort orders by descending priority score.
        2. For each order, pick the vehicle at the same warehouse (or nearest
           home base) with sufficient remaining capacity.
        3. If no vehicle qualifies, the order stays pending.

        Returns
        -------
        int  number of orders newly assigned this call
        """
        pending = sorted(
            [o for o in self.orders.values() if o.status == "pending"],
            key=lambda o: o.priority_score,
            reverse=True,
        )
        assigned_count = 0
        for order in pending:
            vehicle = self._select_vehicle_for(order)
            if vehicle is None:
                continue
            order.assigned_vehicle = vehicle.vehicle_id
            order.status = "assigned"
            vehicle.deliveries_pending.append(order.order_id)
            if vehicle.status == VehicleStatus.IDLE:
                vehicle.status = VehicleStatus.EN_ROUTE
            assigned_count += 1
        return assigned_count

    def complete_delivery(self, order_id: str, step: int) -> None:
        """Mark a delivery as completed and update vehicle state."""
        order = self.orders.get(order_id)
        if order is None:
            return
        order.status = "delivered"
        order.completed_at_step = step

        vehicle = self.vehicles.get(order.assigned_vehicle or "")
        if vehicle is None:
            return
        if order_id in vehicle.deliveries_pending:
            vehicle.deliveries_pending.remove(order_id)
        vehicle.deliveries_completed += 1
        if not vehicle.deliveries_pending:
            vehicle.status = VehicleStatus.IDLE

    def refuel_vehicle(self, vehicle_id: str) -> None:
        """Refuel a vehicle to full tank."""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle:
            vehicle.fuel_level = self._fleet_cfg.fuel_tank_size
            if vehicle.status == VehicleStatus.OUT_OF_FUEL:
                vehicle.status = VehicleStatus.IDLE

    # ------------------------------------------------------------------
    # KPI helpers
    # ------------------------------------------------------------------

    @property
    def fleet_utilisation(self) -> float:
        """Fraction of vehicles currently active (not idle)."""
        if not self.vehicles:
            return 0.0
        active = sum(
            1 for v in self.vehicles.values() if v.status != VehicleStatus.IDLE
        )
        return round(active / len(self.vehicles), 4)

    @property
    def total_deliveries_completed(self) -> int:
        return sum(v.deliveries_completed for v in self.vehicles.values())

    @property
    def average_fuel_level(self) -> float:
        if not self.vehicles:
            return 0.0
        return round(
            sum(v.fuel_level for v in self.vehicles.values()) / len(self.vehicles), 2
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_fleet(self) -> None:
        cap_min, cap_max = self._fleet_cfg.vehicle_capacity_range
        for i in range(self._fleet_cfg.num_vehicles):
            vehicle_id = f"VH-{i:03d}"
            home = self._warehouses[i % len(self._warehouses)]
            capacity = self._rng.randint(cap_min, cap_max)
            self.vehicles[vehicle_id] = Vehicle(
                vehicle_id=vehicle_id,
                capacity=capacity,
                fuel_level=self._fleet_cfg.fuel_tank_size,
                current_location=home,
                home_base=home,
            )

    # ------------------------------------------------------------------
    # Assignment helpers
    # ------------------------------------------------------------------

    def _select_vehicle_for(self, order: DeliveryOrder) -> Optional[Vehicle]:
        """
        Return the best available vehicle for this order, or None.

        Preference order:
        1. Vehicle stationed at the same warehouse as the order's origin.
        2. Vehicle with highest remaining capacity.
        """
        candidates = [
            v for v in self.vehicles.values()
            if v.is_available and v.available_capacity >= order.payload_units
        ]
        if not candidates:
            return None

        # Prefer vehicle already at the origin warehouse
        at_origin = [v for v in candidates if v.current_location == order.origin]
        pool = at_origin if at_origin else candidates

        return max(pool, key=lambda v: v.available_capacity)
