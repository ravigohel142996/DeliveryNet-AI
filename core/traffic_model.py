"""
core/traffic_model.py
---------------------
Bridges the city network and the ML traffic predictor.

Responsibilities:
- Maintain the current simulation time and weather state.
- Use TrafficPredictor to refresh edge weights on every time step.
- Expose traffic-density data for heatmap visualisation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, TrafficConfig
from core.city_network import CityNetwork
from models.traffic_predictor import TrafficPredictor


class TrafficModel:
    """
    Stateful traffic model that updates edge weights in the city network
    using predicted congestion values.

    Parameters
    ----------
    network         : CityNetwork   the underlying graph to update
    traffic_cfg     : TrafficConfig configuration constants
    """

    def __init__(
        self,
        network: CityNetwork,
        traffic_cfg: TrafficConfig = DEFAULT_CONFIG.traffic,
    ) -> None:
        self._network = network
        self._cfg = traffic_cfg
        self._predictor = TrafficPredictor(traffic_cfg)
        self._rng = np.random.default_rng(traffic_cfg.random_seed)

        # Current simulation state
        self.current_hour: int = 8          # default: morning peak
        self.current_weather: str = "clear"
        self.traffic_density_map: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, hour: int, weather: str) -> None:
        """
        Refresh all edge travel times using the ML predictor.

        Parameters
        ----------
        hour    : simulation hour 0-23
        weather : current weather string
        """
        self.current_hour = hour
        self.current_weather = weather
        self._update_edge_weights()
        self._refresh_density_map()

    def get_edge_traffic_dataframe(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of edges with their traffic attributes.
        Useful for heatmap and analytics visualisations.
        """
        rows = []
        for u, v, data in self._network.graph.edges(data=True):
            rows.append(
                {
                    "source": u,
                    "target": v,
                    "road_type": data["road_type"],
                    "distance_km": data["distance_km"],
                    "congestion": data["congestion"],
                    "travel_time": data["travel_time"],
                    "traffic_density": self.traffic_density_map.get(
                        f"{u}-{v}", 0.5
                    ),
                }
            )
        return pd.DataFrame(rows)

    @property
    def model_metrics(self) -> Dict[str, float]:
        """Return ML model performance metrics."""
        return {
            "train_mae": round(self._predictor.train_mae, 4),
            "test_mae": round(self._predictor.test_mae, 4),
            "test_r2": round(self._predictor.test_r2, 4),
        }

    @property
    def feature_importances(self) -> pd.Series:
        return self._predictor.feature_importances

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_edge_weights(self) -> None:
        """Predict congestion for all edges in a single batch call."""
        edges = list(self._network.graph.edges(data=True))
        if not edges:
            return

        weather_enc = self._predictor._safe_encode(
            self._predictor._weather_encoder, self.current_weather
        )

        rows = []
        for u, v, data in edges:
            road_enc = self._predictor._safe_encode(
                self._predictor._road_encoder, data["road_type"]
            )
            density = self.traffic_density_map.get(f"{u}-{v}", 0.5)
            rows.append(
                {
                    "time_of_day": self.current_hour,
                    "weather_encoded": weather_enc,
                    "road_type_encoded": road_enc,
                    "traffic_density": density,
                }
            )

        feature_df = pd.DataFrame(rows)
        congestion_values = self._predictor.predict_batch(feature_df)

        for (u, v, data), congestion in zip(edges, congestion_values):
            clamped = max(1.0, float(congestion))
            data["congestion"] = round(clamped, 3)
            data["travel_time"] = round(data["base_time_min"] * clamped, 3)

    def _refresh_density_map(self) -> None:
        """Assign pseudo-random traffic density per edge (simulates sensor data)."""
        for u, v in self._network.graph.edges():
            key = f"{u}-{v}"
            # Density fluctuates around a base influenced by hour
            if 7 <= self.current_hour <= 9 or 16 <= self.current_hour <= 19:
                base_density = self._rng.uniform(0.5, 1.0)
            elif 22 <= self.current_hour or self.current_hour <= 5:
                base_density = self._rng.uniform(0.0, 0.25)
            else:
                base_density = self._rng.uniform(0.2, 0.7)
            self.traffic_density_map[key] = round(float(base_density), 3)
