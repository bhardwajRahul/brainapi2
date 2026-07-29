#!/usr/bin/env python3
"""Classify wrong LoCoMo answers into generation vs retrieval failure modes."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from locomo.config import CATEGORY_NAMES, RUNS_DIR
from locomo.metrics import evidence_coverage, gold_in_context, tokenize

_RELATIVE_DATE = re.compile(
    r"\b(week of|since \d{4}|sunday before|saturday before|the day before)\b",
    re.I,
)
_ABSTAIN = re.compile(
    r"not mentioned|no information|unknown|cannot (tell|determine)",
    re.I,
)


def _load_latest(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        key = (str(row.get("sample_id")), int(row.get("qa_index") or 0))
        latest[key] = row
    return list(latest.values())


def classify(row: dict[str, Any]) -> str:
    cat = int(row.get("category") or 0)
    gold = str(row.get("gold") or "")
    pred = str(row.get("prediction") or "")
    reason = str(row.get("judge_reason") or "").lower()
    answerable = gold_in_context(row)
    coverage = evidence_coverage(row, "combined")

    if not answerable and coverage in {None, "none", "partial"}:
        return "missing_evidence"
    if _ABSTAIN.search(pred) and gold.strip():
        return "present_but_unused"
    if cat == 3:
        return "open_domain_inference"
    if cat == 2 and (
        _RELATIVE_DATE.search(gold)
        or "date" in reason
        or "temporal" in reason
        or "phrasing" in reason
    ):
        return "temporal_format"
    if cat == 1:
        return "multi_hop_composition"
    if answerable:
        gold_toks = set(tokenize(gold))
        pred_toks = set(tokenize(pred))
        if gold_toks and len(gold_toks & pred_toks) / len(gold_toks) >= 0.4:
            return "phrasing"
        return "present_but_unused"
    return "missing_evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="phase-d1-paths-a",
        help="Run id under benchmarks/runs/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown output path",
    )
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.exists():
        raise SystemExit(f"missing {answers_path}")

    rows = _load_latest(answers_path)
    wrong = [
        r
        for r in rows
        if not r.get("judge_correct") and int(r.get("category") or 0) != 5
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wrong:
        by_class[classify(row)].append(row)

    counts = Counter({k: len(v) for k, v in by_class.items()})
    n_wrong = len(wrong)
    gen_classes = {
        "present_but_unused",
        "phrasing",
        "open_domain_inference",
        "temporal_format",
        "multi_hop_composition",
    }
    gen_n = sum(counts[c] for c in gen_classes)
    gen_share = (gen_n / n_wrong) if n_wrong else 0.0
    n_total = len([r for r in rows if int(r.get("category") or 0) != 5])

    lines: list[str] = []
    lines.append(f"# SOTA failure taxonomy — `{args.run}`")
    lines.append("")
    lines.append(f"Non-adversarial scored: **{n_total}**; wrong: **{n_wrong}**.")
    lines.append(
        f"Generation-side share (expected ≥70%): **{gen_share:.0%}** "
        f"({gen_n}/{n_wrong})."
    )
    lines.append("")
    lines.append("| Class | Count | Est. headline pp if all fixed |")
    lines.append("| --- | ---: | ---: |")
    for cls, n in counts.most_common():
        pp = 100.0 * n / n_total if n_total else 0.0
        lines.append(f"| `{cls}` | {n} | ~{pp:.1f} |")
    lines.append("")
    lines.append("## Top examples (≤20)")
    lines.append("")
    shown = 0
    for cls, items in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        for row in items[: max(1, 20 // max(1, len(by_class)))]:
            if shown >= 20:
                break
            cat = CATEGORY_NAMES.get(int(row.get("category") or 0), "?")
            lines.append(
                f"- **{cls}** / {cat}: Q={row.get('question')!r} "
                f"gold={row.get('gold')!r} pred={row.get('prediction')!r} "
                f"reason={row.get('judge_reason')!r}"
            )
            shown += 1
        if shown >= 20:
            break

    out = args.out or (RUNS_DIR / "sota-failure-taxonomy.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(f"generation_share={gen_share:.3f} wrong={n_wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
