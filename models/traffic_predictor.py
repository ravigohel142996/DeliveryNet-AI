"""
models/traffic_predictor.py
---------------------------
Trains and exposes a Random Forest Regressor that predicts travel-time
delay given contextual road / weather / time features.

The model is trained once at instantiation on a synthetically generated
dataset that reflects realistic traffic patterns.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import TrafficConfig, DEFAULT_CONFIG


class TrafficPredictor:
    """
    Random Forest model for predicting travel-time multipliers.

    Features
    --------
    time_of_day     : int       hour 0-23
    weather_encoded : int       label-encoded weather string
    road_type_encoded: int      label-encoded road-type string
    traffic_density : float     normalised 0-1

    Target
    ------
    congestion_factor : float   ≥ 1.0  (multiplicative delay)
    """

    _FEATURE_COLUMNS: List[str] = [
        "time_of_day",
        "weather_encoded",
        "road_type_encoded",
        "traffic_density",
    ]

    def __init__(self, cfg: TrafficConfig = DEFAULT_CONFIG.traffic) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.random_seed)
        self._weather_encoder = LabelEncoder().fit(list(cfg.weather_conditions))
        self._road_encoder = LabelEncoder().fit(list(cfg.road_types))
        self._model: Optional[RandomForestRegressor] = None
        self.train_mae: float = 0.0
        self.test_mae: float = 0.0
        self.test_r2: float = 0.0
        self._train()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        time_of_day: int,
        weather: str,
        road_type: str,
        traffic_density: float,
    ) -> float:
        """
        Return a predicted congestion multiplier for the given inputs.

        Parameters
        ----------
        time_of_day     : 0-23
        weather         : e.g. 'rain'
        road_type       : e.g. 'arterial'
        traffic_density : 0.0-1.0

        Returns
        -------
        float  predicted congestion factor (clamped ≥ 1.0)
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")

        weather_enc = self._safe_encode(self._weather_encoder, weather)
        road_enc = self._safe_encode(self._road_encoder, road_type)

        features = pd.DataFrame(
            [[time_of_day, weather_enc, road_enc, traffic_density]],
            columns=self._FEATURE_COLUMNS,
        )
        prediction: float = float(self._model.predict(features)[0])
        return max(1.0, round(prediction, 4))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorised prediction over a DataFrame with columns matching
        _FEATURE_COLUMNS (pre-encoded).
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self._model.predict(df[self._FEATURE_COLUMNS])

    @property
    def feature_importances(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        return pd.Series(
            self._model.feature_importances_,
            index=self._FEATURE_COLUMNS,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _train(self) -> None:
        df = self._generate_synthetic_data()
        X = df[self._FEATURE_COLUMNS]
        y = df["congestion_factor"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self._cfg.random_seed
        )

        self._model = RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=self._cfg.random_seed,
        )
        self._model.fit(X_train, y_train)

        train_pred = self._model.predict(X_train)
        test_pred = self._model.predict(X_test)
        self.train_mae = float(mean_absolute_error(y_train, train_pred))
        self.test_mae = float(mean_absolute_error(y_test, test_pred))
        self.test_r2 = float(r2_score(y_test, test_pred))

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Build a labelled dataset with realistic congestion patterns.
        Peak hours and adverse weather inflate the congestion factor.
        """
        n = self._cfg.training_samples
        hours = self._rng.integers(0, 24, n)
        weather_idx = self._rng.integers(0, len(self._cfg.weather_conditions), n)
        road_idx = self._rng.integers(0, len(self._cfg.road_types), n)
        density = self._rng.uniform(0.0, 1.0, n)

        weather_strs = np.array(self._cfg.weather_conditions)[weather_idx]
        road_strs = np.array(self._cfg.road_types)[road_idx]

        # Build congestion label with domain-informed rules + noise
        congestion = np.ones(n)

        # Time-of-day effect
        peak_mask = ((hours >= 7) & (hours <= 9)) | ((hours >= 16) & (hours <= 19))
        midday_mask = (hours >= 11) & (hours <= 13)
        night_mask = (hours >= 22) | (hours <= 5)
        congestion[peak_mask] += 0.8
        congestion[midday_mask] += 0.3
        congestion[night_mask] -= 0.2

        # Weather effect
        weather_impact = {"clear": 0.0, "rain": 0.3, "fog": 0.5, "snow": 1.0}
        for wth, impact in weather_impact.items():
            congestion[weather_strs == wth] += impact

        # Traffic density (continuous)
        congestion += density * 0.9

        # Road-type sensitivity (highways handle traffic better)
        sensitivity = {"highway": 0.6, "arterial": 0.8, "local": 1.0, "residential": 1.1}
        for rt, s in sensitivity.items():
            mask = road_strs == rt
            excess = congestion[mask] - 1.0
            congestion[mask] = 1.0 + excess * s

        # Gaussian noise for realism
        noise = self._rng.normal(0.0, 0.08, n)
        congestion = np.clip(congestion + noise, 1.0, 3.5)

        return pd.DataFrame(
            {
                "time_of_day": hours,
                "weather_encoded": self._weather_encoder.transform(weather_strs),
                "road_type_encoded": self._road_encoder.transform(road_strs),
                "traffic_density": density,
                "congestion_factor": congestion,
            }
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_encode(encoder: LabelEncoder, value: str) -> int:
        classes = list(encoder.classes_)
        if value not in classes:
            return 0  # fallback to first class
        return int(encoder.transform([value])[0])
