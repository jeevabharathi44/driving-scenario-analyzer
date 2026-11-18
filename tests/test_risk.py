import numpy as np

from scenario_analyzer.risk import FEATURES, WEIGHTS, feature_vector, risk_level, risk_score

CALM = {"road_type": "highway", "weather": "clear", "time_of_day": "day",
        "ego_maneuver": "cruise", "actors": [], "speed_kmh": 30}
HARSH = {"road_type": "intersection", "weather": "fog", "time_of_day": "night",
         "ego_maneuver": "turn_left", "actors": ["pedestrian", "cyclist"], "speed_kmh": 80}


def test_weights_match_features():
    assert len(FEATURES) == len(WEIGHTS)


def test_feature_vector_shape_and_speed_factor():
    vec = feature_vector(CALM)
    assert vec.shape == (len(FEATURES),)
    assert np.isclose(vec[FEATURES.index("speed_factor")], 0.25)


def test_harsh_scenario_scores_higher_than_calm():
    assert risk_score(HARSH) > risk_score(CALM)


def test_score_is_clipped_to_unit_interval():
    extreme = {**HARSH, "weather": "snow", "actors": ["pedestrian", "cyclist", "animal", "truck", "vehicle"],
               "speed_kmh": 300}
    assert 0.0 <= risk_score(extreme) <= 1.0
    assert risk_score(extreme) == 1.0


def test_missing_speed_counts_as_zero_speed_factor():
    assert feature_vector({**CALM, "speed_kmh": None})[FEATURES.index("speed_factor")] == 0.0


def test_levels():
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.9) == "high"
