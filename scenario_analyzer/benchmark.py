"""Measure extraction accuracy against a hand-labelled benchmark file."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

FIELDS = ["road_type", "weather", "time_of_day", "ego_maneuver", "actors", "speed_kmh", "risk_level"]


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def evaluate(predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-field accuracy. Both frames must be in the same row order."""
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same number of rows")
    rows = []
    for field in FIELDS:
        p, t = _norm(predicted[field]), _norm(truth[field])
        rows.append({"field": field, "accuracy": round(float(accuracy_score(t, p)), 3), "n": len(t)})
    table = pd.DataFrame(rows)
    overall = {"field": "overall (mean)", "accuracy": round(float(table["accuracy"].mean()), 3), "n": len(truth)}
    return pd.concat([table, pd.DataFrame([overall])], ignore_index=True)


def risk_report(predicted: pd.DataFrame, truth: pd.DataFrame) -> str:
    return classification_report(
        _norm(truth["risk_level"]), _norm(predicted["risk_level"]), zero_division=0
    )
