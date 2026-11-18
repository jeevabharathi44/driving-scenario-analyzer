import pytest

from scenario_analyzer.schema import ValidationError, validate

GOOD = {
    "road_type": "Highway", "weather": "rain", "time_of_day": "night",
    "ego_maneuver": "overtake", "actors": ["Truck", "truck", "cyclist"], "speed_kmh": 90.4,
}


def test_valid_scenario_is_cleaned():
    out = validate(GOOD)
    assert out["road_type"] == "highway"
    assert out["actors"] == ["cyclist", "truck"]  # lower-cased, de-duplicated, sorted
    assert out["speed_kmh"] == 90


def test_speed_can_be_null():
    assert validate({**GOOD, "speed_kmh": None})["speed_kmh"] is None


@pytest.mark.parametrize("field,value", [
    ("road_type", "moon"), ("weather", "hail"), ("time_of_day", "noon"), ("ego_maneuver", "fly"),
])
def test_bad_enum_values_rejected(field, value):
    with pytest.raises(ValidationError):
        validate({**GOOD, field: value})


def test_missing_key_rejected():
    bad = dict(GOOD)
    del bad["weather"]
    with pytest.raises(ValidationError, match="missing"):
        validate(bad)


@pytest.mark.parametrize("speed", [-5, 999, "fast", True])
def test_bad_speed_rejected(speed):
    with pytest.raises(ValidationError):
        validate({**GOOD, "speed_kmh": speed})


def test_unknown_actor_rejected():
    with pytest.raises(ValidationError, match="unknown actors"):
        validate({**GOOD, "actors": ["dragon"]})
