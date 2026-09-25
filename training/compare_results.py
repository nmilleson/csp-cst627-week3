"""Print a markdown comparison table from evaluate.py summary files.

Usage:
    python training/compare_results.py eval_results/base.summary.json eval_results/sft.summary.json eval_results/dpo.summary.json
"""

import argparse
import json
from pathlib import Path

METRICS = [
    "valid_json_rate", "has_all_fields_rate", "category_accuracy",
    "priority_accuracy", "exact_match_rate", "seconds_per_example",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in args.summaries]

    print("| metric | " + " | ".join(r["label"] for r in runs) + " |")
    print("|---" * (len(runs) + 1) + "|")
    for metric in METRICS:
        values = " | ".join(
            f"{r[metric]:.3f}" if r.get(metric) is not None else "-" for r in runs
        )
        print(f"| {metric} | {values} |")


if __name__ == "__main__":
    main()
