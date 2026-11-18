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

## Results

### Rule-based extractor (`--extractor rule`)

| field | accuracy |
|---|---|
| road_type | 1.00 |
| weather | 1.00 |
| time_of_day | 1.00 |
| ego_maneuver | 1.00 |
| actors | 1.00 |
| speed_kmh | 1.00 |
| risk_level | 0.85 |
| **overall (mean)** | **0.98** |

### LLM extractor (`--extractor groq`, model: `qwen/qwen3.8-27b` via Groq API)

| field | accuracy |
|---|---|
| road_type | 1.00 |
| weather | 1.00 |
| time_of_day | 1.00 |
| ego_maneuver | 1.00 |
| actors | 1.00 |
| speed_kmh | 1.00 |
| risk_level | 0.85 |
| **overall (mean)** | **0.98** |

Risk level breakdown (LLM, 20 scenarios):

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| high | 0.80 | 0.89 | 0.84 | 9 |
| low | 1.00 | 1.00 | 1.00 | 5 |
| medium | 0.80 | 0.67 | 0.73 | 6 |
| **accuracy** | | | **0.85** | 20 |

**Interpretation:** Both extractors score identically on this benchmark (0.98 mean). This is expected: the 20-scenario benchmark was written by the author in a style that keyword rules handle well, and a capable LLM learns the same patterns. The LLM's real advantage would appear on messier, real-world text where keyword rules break down. A larger, more varied benchmark would give a more honest comparison.

## Limitations and next steps

- The benchmark is tiny; a larger, more varied labelled set would give a more honest comparison.
- The risk score is a hand-set heuristic. With real labelled outcomes, the weights could be fitted (for example with scikit-learn logistic regression).
- Ideas: compare several models, add retrieval over a scenario library, expose the pipeline as a FastAPI endpoint.
