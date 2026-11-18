"""Four ways to turn scenario text into a validated dict.

* RuleBasedExtractor: keyword/regex baseline, works offline.
* LLMExtractor: calls the Anthropic API, asks for JSON, validates it, retries once.
* GeminiExtractor: calls the Google Gemini API (free tier), same contract as LLMExtractor.
* GroqExtractor: calls the Groq API (free tier, LLaMA-3), OpenAI-compatible interface.
"""

from __future__ import annotations

import json
import os
import re

from .schema import validate

SYSTEM_PROMPT = """You convert short driving-scenario descriptions into JSON.
Return ONLY one JSON object, with no markdown and no commentary, using exactly these keys:
- road_type: one of highway, urban, rural, intersection, parking_lot
- weather: one of clear, rain, fog, snow (use clear if not mentioned)
- time_of_day: one of day, dusk, night (use day if not mentioned)
- ego_maneuver: one of cruise, brake, lane_change, overtake, turn_left, turn_right
- actors: list of other road users present, each one of pedestrian, cyclist, vehicle, truck, animal
  (buses and cars count as vehicle; do not list actors that are explicitly absent; the ego car is not an actor)
- speed_kmh: the ego speed as an integer in km/h, or null if not stated
"""


class ExtractionError(RuntimeError):
    """Raised when an extractor cannot produce a valid scenario."""


# ---------------------------------------------------------------- rule based
_ACTOR_WORDS = {
    "pedestrian": r"pedestrians?|walker",
    "cyclist": r"cyclists?|bicycl\w+",
    "truck": r"trucks?|lorry",
    "vehicle": r"vehicles?|traffic|buses|bus|cars?|taxi",
    "animal": r"animals?|deer|dog|cat|cow",
}


class RuleBasedExtractor:
    """Simple keyword baseline used for tests and for comparison with the LLM."""

    def extract(self, text: str) -> dict:
        t = text.lower()
        # the ego vehicle is not another actor
        t_actors = re.sub(r"ego[ -](vehicle|car)", " ", t)

        data = {
            "road_type": self._road_type(t),
            "weather": self._first(t, [("fog", r"fog"), ("snow", r"snow"), ("rain", r"rain")], "clear"),
            "time_of_day": self._first(t, [("night", r"night"), ("dusk", r"dusk")], "day"),
            "ego_maneuver": self._maneuver(t),
            "actors": self._actors(t_actors),
            "speed_kmh": self._speed(t),
        }
        return validate(data)

    @staticmethod
    def _first(text, rules, default):
        for label, pattern in rules:
            if re.search(pattern, text):
                return label
        return default

    def _road_type(self, t):
        return self._first(
            t,
            [
                ("parking_lot", r"parking"),
                ("intersection", r"intersection|junction"),
                ("highway", r"highway|motorway"),
                ("rural", r"rural|country road"),
                ("urban", r"urban|city|street|crosswalk"),
            ],
            "urban",
        )

    def _maneuver(self, t):
        return self._first(
            t,
            [
                ("overtake", r"overtak"),
                ("lane_change", r"chang\w* lane|lane change"),
                ("turn_left", r"turn\w* left"),
                ("turn_right", r"turn\w* right"),
                ("brake", r"brak"),
            ],
            "cruise",
        )

    @staticmethod
    def _actors(t):
        found = set()
        for actor, pattern in _ACTOR_WORDS.items():
            for match in re.finditer(rf"\b(" + pattern + ")\b", t):
                preceding = " ".join(t[: match.start()].split()[-3:])
                if re.search(r"\b(no|without)\b", preceding):
                    continue  # explicitly absent, e.g. "no pedestrians nearby"
                found.add(actor)
        return sorted(found)

    @staticmethod
    def _speed(t):
        match = re.search(r"(\d+(?:\.\d+)?)" + "\s*km\s*/?\s*h", t)
        return int(float(match.group(1))) if match else None


# ----------------------------------------------------------------------- LLM
def parse_json_response(raw: str) -> dict:
    """Extract a JSON object from a model reply, tolerating markdown fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model reply")
    return json.loads(cleaned[start : end + 1])


class LLMExtractor:
    """Calls an LLM and validates the JSON it returns.

    `client` is injectable so tests can run without network access or an API key.
    """

    def __init__(self, client=None, model: str | None = None, max_retries: int = 1):
        if client is None:
            import anthropic  # imported lazily so the rule-based path needs no key

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client
        self.model = model or os.environ.get("SCENARIO_MODEL", "claude-sonnet-5-5")
        self.max_retries = max_retries

    def extract(self, text: str) -> dict:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            reply = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            raw = "".join(b.text for b in reply.content if getattr(b, "type", "") == "text")
            try:
                return validate(parse_json_response(raw))
            except ValueError as err:  # includes JSONDecodeError and ValidationError
                last_error = err
        raise ExtractionError(f"model output was invalid after retries: {last_error}")


class GeminiExtractor:
    """Calls Google Gemini API (free tier) and validates the JSON it returns.

    Requires the GEMINI_API_KEY environment variable.
    Install: pip install google-generativeai
    """

    def __init__(self, model: str | None = None, max_retries: int = 1):
        from google import genai  # lazy import — uses the new google-genai package

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key at https://aistudio.google.com"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model or os.environ.get("SCENARIO_MODEL", "gemini-2.0-flash")
        self.max_retries = max_retries

    def extract(self, text: str) -> dict:
        from google.genai import types

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=400,
                ),
            )
            raw = response.text
            try:
                return validate(parse_json_response(raw))
            except ValueError as err:
                last_error = err
        raise ExtractionError(f"model output was invalid after retries: {last_error}")


class GroqExtractor:
    """Calls the Groq API (free tier) using LLaMA-3 and validates the JSON it returns.

    Requires the GROQ_API_KEY environment variable.
    Get a free key at https://console.groq.com
    Install: pip install groq
    """

    def __init__(self, model: str | None = None, max_retries: int = 1):
        from groq import Groq  # lazy import

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model or os.environ.get("SCENARIO_MODEL", "qwen/qwen3.8-27b")
        self.max_retries = max_retries

    def extract(self, text: str) -> dict:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            raw = response.choices[0].message.content
            try:
                return validate(parse_json_response(raw))
            except ValueError as err:
                last_error = err
        raise ExtractionError(f"model output was invalid after retries: {last_error}")


def get_extractor(name: str):
    if name == "rule":
        return RuleBasedExtractor()
    if name == "llm":
        return LLMExtractor()
    if name == "gemini":
        return GeminiExtractor()
    if name == "groq":
        return GroqExtractor()
    raise ValueError(f"unknown extractor {name!r} (use 'rule', 'llm', 'gemini', or 'groq')")
