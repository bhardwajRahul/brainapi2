# Synergies / recommend structural tripwire
#
# HTTP-optional local metrics helpers only. Does **not** upsert
# benchmarks/REPORTS.json (LoCoMo / LongMemEval / BEAM).
#
# Usage (from repo root, after retrieving recommendations for a seed):
#   PYTHONPATH=. poetry run python -m benchmarks.synergies.report \
#     --labels PRODUCT,PHONE PRODUCT,CASE CATEGORY --distances 1 2 3
#
# Prefer dedicated brains such as `demorecsys`, never wipe beam1m1clean / locomoconv26*.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.search.synergy_metrics import summarize_recommendation_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural synergies tripwire report")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Space-separated label sets; within a set use commas (e.g. PRODUCT,PHONE)",
    )
    parser.add_argument(
        "--distances",
        nargs="*",
        type=int,
        default=[],
        help="Optional hop distances aligned with --labels",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    label_sets = [item.split(",") for item in args.labels]
    distances = args.distances or [1] * len(label_sets)
    if len(distances) != len(label_sets):
        distances = (distances + [1] * len(label_sets))[: len(label_sets)]

    report = summarize_recommendation_list(label_sets, distances)
    report["suite"] = "synergies-structural"
    report["ledger"] = None
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
