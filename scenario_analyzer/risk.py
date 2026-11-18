"""Heuristic risk score built with NumPy.

This is a transparent, hand-weighted score, not a trained model. It exists so
extracted scenarios can be ranked and bucketed, and so the weights are easy to
discuss and change.
"""

from __future__ import annotations

import numpy as np

# One weight per feature. Order matters: FEATURES and WEIGHTS line up by index.
FEATURES = [
    "weather=rain", "weather=fog", "weather=snow",
    "time=night", "time=dusk",
    "road=intersection", "road=urban", "road=rural", "road=highway",
    "move=overtake", "move=lane_change", "move=turn_left", "move=turn_right", "move=brake",
    "actor=pedestrian", "actor=cyclist", "actor=animal", "actor=truck", "actor=vehicle",
    "speed_factor",
]
WEIGHTS = np.array([
    0.15, 0.25, 0.30,
    0.20, 0.10,
    0.20, 0.10, 0.08, 0.05,
    0.15, 0.10, 0.15, 0.08, 0.12,
    0.20, 0.20, 0.15, 0.08, 0.05,
    0.25,
])
LOW_MAX = 0.35
MEDIUM_MAX = 0.60


def feature_vector(s: dict) -> np.ndarray:
    names = {
        f"weather={s['weather']}", f"time={s['time_of_day']}",
        f"road={s['road_type']}", f"move={s['ego_maneuver']}",
        *[f"actor={a}" for a in s["actors"]],
    }
    vec = np.array([1.0 if f in names else 0.0 for f in FEATURES])
    speed = s.get("speed_kmh")
    vec[FEATURES.index("speed_factor")] = 0.0 if speed is None else min(speed / 120.0, 1.0)
    return vec


def risk_score(s: dict) -> float:
    """Weighted sum of features, clipped to [0, 1]."""
    return float(np.clip(np.dot(feature_vector(s), WEIGHTS), 0.0, 1.0))


def risk_level(score: float) -> str:
    if score < LOW_MAX:
        return "low"
    if score < MEDIUM_MAX:
        return "medium"
    return "high"
