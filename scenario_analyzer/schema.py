"""Schema and validation for structured scenario descriptions."""

from __future__ import annotations

ALLOWED = {
    "road_type": {"highway", "urban", "rural", "intersection", "parking_lot"},
    "weather": {"clear", "rain", "fog", "snow"},
    "time_of_day": {"day", "dusk", "night"},
    "ego_maneuver": {"cruise", "brake", "lane_change", "overtake", "turn_left", "turn_right"},
}
ALLOWED_ACTORS = {"pedestrian", "cyclist", "vehicle", "truck", "animal"}
REQUIRED_KEYS = set(ALLOWED) | {"actors", "speed_kmh"}


class ValidationError(ValueError):
    """Raised when an extracted scenario does not match the schema."""


def validate(data: dict) -> dict:
    """Check a scenario dict and return a cleaned copy.

    Raises ValidationError with a readable message if anything is wrong.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"expected an object, got {type(data).__name__}")

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValidationError(f"missing keys: {sorted(missing)}")

    cleaned: dict = {}
    for key, allowed in ALLOWED.items():
        value = str(data[key]).strip().lower()
        if value not in allowed:
            raise ValidationError(f"{key}={value!r} not in {sorted(allowed)}")
        cleaned[key] = value

    actors = data["actors"]
    if not isinstance(actors, list):
        raise ValidationError("actors must be a list")
    actors = sorted({str(a).strip().lower() for a in actors})
    bad = [a for a in actors if a not in ALLOWED_ACTORS]
    if bad:
        raise ValidationError(f"unknown actors: {bad}")
    cleaned["actors"] = actors

    speed = data["speed_kmh"]
    if speed is None:
        cleaned["speed_kmh"] = None
    else:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValidationError("speed_kmh must be a number or null")
        if speed < 0 or speed > 300:
            raise ValidationError("speed_kmh out of range 0-300")
        cleaned["speed_kmh"] = int(round(speed))

    return cleaned
