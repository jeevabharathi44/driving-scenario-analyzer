import pandas as pd
import pytest

from scenario_analyzer.benchmark import evaluate
from scenario_analyzer.extractors import RuleBasedExtractor
from scenario_analyzer.pipeline import analyse


class BrokenExtractor:
    def extract(self, text):
        raise RuntimeError("boom")


def test_pipeline_adds_columns():
    df = pd.DataFrame({"text": ["The ego car brakes at 40 km/h on an urban street, a pedestrian crosses."]})
    out = analyse(df, RuleBasedExtractor())
    assert {"road_type", "risk_score", "risk_level", "error"} <= set(out.columns)
    assert out.loc[0, "actors"] == "pedestrian" and out.loc[0, "error"] == ""


def test_pipeline_needs_text_column():
    with pytest.raises(ValueError):
        analyse(pd.DataFrame({"foo": [1]}), RuleBasedExtractor())


def test_pipeline_survives_extractor_errors():
    out = analyse(pd.DataFrame({"text": ["a", "b"]}), BrokenExtractor())
    assert list(out["error"]) == ["boom", "boom"]


def test_pipeline_keeps_label_columns_without_clashing():
    df = pd.read_csv("data/benchmark.csv").head(3)
    out = analyse(df, RuleBasedExtractor())
    assert "label_road_type" in out.columns and "road_type" in out.columns


def test_benchmark_perfect_and_imperfect():
    truth = pd.read_csv("data/benchmark.csv")
    perfect = truth.copy()
    assert evaluate(perfect, truth).iloc[-1]["accuracy"] == 1.0

    wrong = truth.copy()
    wrong.loc[0, "weather"] = "snow"
    row = evaluate(wrong, truth).set_index("field").loc["weather"]
    assert row["accuracy"] == pytest.approx(0.95)


def test_benchmark_row_count_mismatch():
    truth = pd.read_csv("data/benchmark.csv")
    with pytest.raises(ValueError):
        evaluate(truth.head(5), truth)


def test_rule_baseline_on_benchmark_is_reasonable():
    truth = pd.read_csv("data/benchmark.csv")
    pred = analyse(truth[["text"]], RuleBasedExtractor())
    scores = evaluate(pred, truth).set_index("field")["accuracy"]
    assert scores["road_type"] >= 0.9 and scores["weather"] >= 0.9
