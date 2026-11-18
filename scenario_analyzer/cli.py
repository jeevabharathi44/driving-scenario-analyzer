from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import evaluate, risk_report
from .extractors import get_extractor
from .pipeline import analyse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scenario_analyzer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="extract structured fields from a CSV with a 'text' column")
    run.add_argument("--input", default="data/benchmark.csv")
    run.add_argument("--output", default="outputs/results.csv")
    run.add_argument("--extractor", choices=["rule", "llm", "gemini", "groq"], default="rule")

    bench = sub.add_parser("benchmark", help="score an extractor against labelled data")
    bench.add_argument("--input", default="data/benchmark.csv")
    bench.add_argument("--extractor", choices=["rule", "llm", "gemini", "groq"], default="rule")

    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    result = analyse(df, get_extractor(args.extractor))

    if args.command == "run":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        failed = int((result["error"] != "").sum())
        print(f"Wrote {len(result)} rows to {args.output} ({failed} failed)")
        print(result["risk_level"].value_counts().to_string())
    else:
        print(f"Extractor: {args.extractor}  |  scenarios: {len(df)}\n")
        print(evaluate(result, df).to_string(index=False))
        print("\nRisk level report:\n" + risk_report(result, df))


if __name__ == "__main__":
    main()
