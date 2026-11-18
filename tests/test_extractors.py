import json
from types import SimpleNamespace

import pytest

from scenario_analyzer.extractors import (
    ExtractionError, LLMExtractor, RuleBasedExtractor, parse_json_response,
)

VALID = {
    "road_type": "urban", "weather": "clear", "time_of_day": "day",
    "ego_maneuver": "brake", "actors": ["pedestrian"], "speed_kmh": 40,
}


class FakeClient:
    """Stands in for the Anthropic client: returns queued replies, records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        text = self.replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_rule_based_basic():
    s = RuleBasedExtractor().extract(
        "At night in heavy rain the ego car overtakes a truck on the highway at 90 km/h."
    )
    assert s["road_type"] == "highway" and s["weather"] == "rain"
    assert s["time_of_day"] == "night" and s["ego_maneuver"] == "overtake"
    assert s["actors"] == ["truck"] and s["speed_kmh"] == 90


def test_rule_based_ignores_ego_vehicle_and_negations():
    s = RuleBasedExtractor().extract(
        "The ego vehicle turns right at an intersection at 20 km/h, with no pedestrians nearby."
    )
    assert s["actors"] == []
    assert s["ego_maneuver"] == "turn_right"


def test_rule_based_defaults_when_unspecified():
    s = RuleBasedExtractor().extract("The ego car drives along.")
    assert (s["weather"], s["time_of_day"], s["ego_maneuver"], s["speed_kmh"]) == ("clear", "day", "cruise", None)


def test_parse_json_handles_markdown_fences():
    raw = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_json_response(raw)["road_type"] == "urban"


def test_parse_json_without_object_fails():
    with pytest.raises(ValueError):
        parse_json_response("sorry, I cannot help")


def test_llm_extractor_returns_validated_dict():
    client = FakeClient([json.dumps(VALID)])
    out = LLMExtractor(client=client, model="test-model").extract("some text")
    assert out["ego_maneuver"] == "brake" and client.calls == 1


def test_llm_extractor_retries_once_after_invalid_output():
    client = FakeClient(["not json at all", json.dumps(VALID)])
    out = LLMExtractor(client=client, model="test-model").extract("some text")
    assert out["road_type"] == "urban" and client.calls == 2


def test_llm_extractor_gives_up_after_retries():
    bad = json.dumps({**VALID, "weather": "hail"})
    client = FakeClient([bad, bad])
    with pytest.raises(ExtractionError):
        LLMExtractor(client=client, model="test-model").extract("some text")
    assert client.calls == 2
