# Driving Scenario Analyzer

A small Python tool that turns free-text driving scenarios (for example, *"At night in heavy rain the ego car overtakes a truck on the highway at 90 km/h"*) into structured, validated data, scores their risk, and measures how accurate the extraction is.

It is inspired by scenario-based testing of autonomous driving systems, where scenarios have to be described in a machine-readable form before they can be analysed or simulated.

## What it does

1. **Extracts** structured fields from text: road type, weather, time of day, ego maneuver, other actors, ego speed.
   - `rule`: a keyword/regex baseline that works offline.
   - `llm`: calls an LLM API (prompt design, JSON output, validation, one retry on bad output).
2. **Validates** every result against a schema (`scenario_analyzer/schema.py`), so bad model output is caught instead of silently passed on.
3. **Scores risk** with a transparent weighted heuristic built with NumPy (`risk.py`). It is hand-weighted, not a trained model.
4. **Benchmarks** an extractor against a hand-labelled CSV with pandas and scikit-learn (per-field accuracy and a risk-level report).
5. **Tests** everything with pytest. The LLM path is tested with a fake client, so tests need no network or API key.

## Project layout

```
scenario_analyzer/
  schema.py       allowed values + validation
  extractors.py   RuleBasedExtractor, LLMExtractor (+ JSON parsing, retry)
  risk.py         NumPy risk score and low/medium/high levels
  pipeline.py     run an extractor over a CSV with pandas
  benchmark.py    accuracy per field with scikit-learn
  cli.py          command line interface
data/benchmark.csv   20 hand-labelled scenarios
tests/               pytest suite (33 tests)
```

## Setup

```bash
git clone https://github.com/jeevabharathi44/driving-scenario-analyzer.git
cd driving-scenario-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Offline baseline (no API key needed):

```bash
python -m scenario_analyzer benchmark --extractor rule
python -m scenario_analyzer run --extractor rule --input data/benchmark.csv --output outputs/results.csv
```

With an LLM (needs an API key; see `.env.example`):

```bash
export ANTHROPIC_API_KEY=your-key-here      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key-here"
python -m scenario_analyzer benchmark --extractor llm
```

Run the tests:

```bash
python -m pytest
```

## Results so far

Rule-based baseline on the 20-scenario benchmark (`python -m scenario_analyzer benchmark --extractor rule`):

| field | accuracy |
|---|---|
| road_type, weather, time_of_day, ego_maneuver, actors, speed_kmh | 1.00 |
| risk_level (from the heuristic score) | 0.85 |
| overall (mean) | 0.98 |

Please read these numbers with care. The benchmark is small and was written by the author in a style that keyword rules handle well, so the baseline looks better than it would on messy real-world text. The LLM extractor has not been scored here yet. Run it with your own key and add the numbers below.

LLM extractor result: _to be added after running `--extractor llm`_

## Limitations and next steps

- The benchmark is tiny; a larger, more varied labelled set would give a more honest comparison.
- The risk score is a hand-set heuristic. With real labelled outcomes, the weights could be fitted (for example with scikit-learn logistic regression).
- Ideas: compare several models, add retrieval over a scenario library, expose the pipeline as a FastAPI endpoint.
