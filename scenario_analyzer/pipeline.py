"""Run an extractor over a table of scenarios and add risk columns."""

from __future__ import annotations

import pandas as pd

from .risk import risk_level, risk_score

RESULT_COLUMNS = [
    "road_type", "weather", "time_of_day", "ego_maneuver", "actors", "speed_kmh",
    "risk_score", "risk_level",
]


def analyse(df: pd.DataFrame, extractor) -> pd.DataFrame:
    """Return df plus extracted fields and risk columns.

    Rows the extractor cannot handle get an `error` message instead of crashing the run.
    """
    if "text" not in df.columns:
        raise ValueError("input needs a 'text' column")

    rows = []
    for text in df["text"]:
        try:
            s = extractor.extract(str(text))
            score = risk_score(s)
            rows.append({**s, "actors": "|".join(s["actors"]),
                         "risk_score": round(score, 3), "risk_level": risk_level(score), "error": ""})
        except Exception as err:  # keep batch runs alive; report per-row failures
            rows.append({"error": str(err)})

    out = pd.DataFrame(rows, columns=RESULT_COLUMNS + ["error"])
    # If the input already has labelled columns with the same names, keep them as label_<name>
    clash = set(df.columns) & set(out.columns)
    base = df.rename(columns={c: f"label_{c}" for c in clash}).reset_index(drop=True)
    return pd.concat([base, out], axis=1)
