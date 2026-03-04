"""
utils/helpers.py
----------------
Shared utility functions used across the DeliveryNet AI codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def format_duration(minutes: float) -> str:
    """
    Convert a float number of minutes into a human-readable string.

    Examples
    --------
    >>> format_duration(75.5)
    '1h 15m'
    >>> format_duration(45.0)
    '45m'
    """
    if minutes == float("inf"):
        return "N/A"
    total_minutes = int(round(minutes))
    hours, mins = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def format_distance(km: float) -> str:
    """
    Format a distance in km for display.

    Examples
    --------
    >>> format_distance(1.4)
    '1.40 km'
    >>> format_distance(0.35)
    '350 m'
    """
    if km == float("inf"):
        return "N/A"
    if km >= 1.0:
        return f"{km:.2f} km"
    return f"{km * 1000:.0f} m"


def pct(value: float) -> str:
    """Format a 0-1 fraction as a percentage string."""
    return f"{value * 100:.1f}%"


def colour_for_congestion(factor: float) -> str:
    """
    Map a congestion factor (1.0–3.5) to a hex colour for visualisation.

    Returns a colour on the green→yellow→red spectrum.
    """
    # Normalise to 0-1
    normalised = min(max((factor - 1.0) / 2.5, 0.0), 1.0)
    if normalised < 0.5:
        r = int(255 * normalised * 2)
        g = 255
    else:
        r = 255
        g = int(255 * (1.0 - (normalised - 0.5) * 2))
    return f"#{r:02x}{g:02x}00"


def summarise_orders(orders: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a summary DataFrame from an order dictionary for display.

    Parameters
    ----------
    orders : dict  {order_id: DeliveryOrder}

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        {
            "Order ID": oid,
            "Origin": o.origin,
            "Destination": o.destination,
            "Priority": o.priority,
            "Status": o.status,
            "Vehicle": o.assigned_vehicle or "—",
            "Payload": o.payload_units,
        }
        for oid, o in orders.items()
    ]
    return pd.DataFrame(rows)


def summarise_fleet(vehicles: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a summary DataFrame from the vehicle dictionary for display.

    Parameters
    ----------
    vehicles : dict  {vehicle_id: Vehicle}

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        {
            "Vehicle ID": vid,
            "Status": v.status.name,
            "Location": v.current_location,
            "Capacity": v.capacity,
            "Available": v.available_capacity,
            "Fuel %": v.fuel_percentage,
            "Deliveries": v.deliveries_completed,
            "Distance (km)": round(v.total_distance_km, 2),
        }
        for vid, v in vehicles.items()
    ]
    return pd.DataFrame(rows)
