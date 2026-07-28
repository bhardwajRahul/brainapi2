from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from locomo.config import CATEGORY_NAMES
from locomo.ingest import load_jsonl
from locomo.metrics import aggregate_answers

console = Console()


def _latest_ingest_rows(ingest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last status per sample/unit (resume appends retries)."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ingest_rows:
        key = (
            str(row.get("sample_id") or ""),
            str(row.get("unit_id") or row.get("session_key") or ""),
        )
        latest[key] = row
    return list(latest.values())


def build_report(run_dir: Path) -> dict[str, Any]:
    answers_path = run_dir / "answers.jsonl"
    ingest_path = run_dir / "ingest.jsonl"
    manifest_path = run_dir / "manifest.json"

    answers = [r for r in load_jsonl(answers_path) if not r.get("error")]
    ingest_rows = load_jsonl(ingest_path)
    ingest_latest = _latest_ingest_rows(ingest_rows)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metrics = aggregate_answers(answers)
    ingest_summary = {
        "n": len(ingest_latest),
        "completed": sum(1 for r in ingest_latest if r.get("status") == "completed"),
        "partial_failed": sum(
            1 for r in ingest_latest if r.get("status") == "partial_failed"
        ),
        "failed": sum(1 for r in ingest_latest if r.get("status") == "failed"),
        "dry_run": sum(1 for r in ingest_latest if r.get("status") == "dry_run"),
        "attempts": len(ingest_rows),
    }

    report = {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "ingest": ingest_summary,
        "metrics": metrics,
        "warnings": [],
    }
    if ingest_summary["partial_failed"] or ingest_summary["failed"]:
        report["warnings"].append(
            "Some ingest units failed or partially failed; scores may be skewed."
        )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    manifest = report.get("manifest") or {}
    lines = [
        "# LoCoMo BrainAPI Benchmark Report",
        "",
        f"- Run dir: `{report['run_dir']}`",
        f"- Created: {manifest.get('created_at', 'n/a')}",
        f"- Granularity: {manifest.get('granularity', 'n/a')}",
        f"- Answer model: {manifest.get('answer_model', 'n/a')}",
        f"- Judge model: {manifest.get('judge_model', 'n/a')}",
        f"- Samples: {', '.join(manifest.get('samples', []) or ['n/a'])}",
        "",
        "## Headline",
        "",
        f"- Judge accuracy (excl. adversarial / cat 5): "
        f"**{_pct(metrics.get('headline_judge_accuracy'))}** "
        f"{_ci(metrics.get('headline_judge_accuracy_ci95'))} "
        f"(n={metrics.get('n_non_adversarial', 0)})",
        f"- Judge accuracy (all categories): "
        f"**{_pct(metrics.get('overall_judge_accuracy'))}** "
        f"{_ci(metrics.get('overall_judge_accuracy_ci95'))} "
        f"(n={metrics.get('n_total', 0)})",
        f"- Mean F1 (excl. adversarial): "
        f"**{_num(metrics.get('mean_f1_non_adversarial'))}**",
        f"- Mean BLEU-1 (excl. adversarial): "
        f"**{_num(metrics.get('mean_bleu1_non_adversarial'))}**",
        "",
        "## Per-category",
        "",
        "| Category | Name | N | Judge Acc | 95% CI | Mean F1 | Mean BLEU-1 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for cat, bucket in (metrics.get("by_category") or {}).items():
        name = CATEGORY_NAMES.get(int(cat), "unknown")
        lines.append(
            f"| {cat} | {name} | {bucket.get('n', 0)} | "
            f"{_pct(bucket.get('judge_accuracy'))} | "
            f"{_ci(bucket.get('judge_accuracy_ci95'))} | "
            f"{_num(bucket.get('mean_f1'))} | "
            f"{_num(bucket.get('mean_bleu1'))} |"
        )

    lat = metrics.get("retrieval_latency_ms") or {}
    lines.extend(
        [
            "",
            "## Latency & tokens",
            "",
            f"- Retrieval latency p50: {_num(lat.get('p50'))} ms",
            f"- Retrieval latency p95: {_num(lat.get('p95'))} ms",
            f"- Retrieval latency mean: {_num(lat.get('mean'))} ms",
            f"- Total LLM tokens (answer+judge): {metrics.get('total_llm_tokens', 0)}",
            "",
            "## Ingest",
            "",
            f"- Units: {report['ingest'].get('n', 0)}",
            f"- Completed: {report['ingest'].get('completed', 0)}",
            f"- Partial failed: {report['ingest'].get('partial_failed', 0)}",
            f"- Failed: {report['ingest'].get('failed', 0)}",
            *(
                [f"- Attempts (incl. retries): {report['ingest'].get('attempts')}"]
                if report["ingest"].get("attempts")
                and report["ingest"].get("attempts") != report["ingest"].get("n")
                else []
            ),
            "",
        ]
    )
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path) -> dict[str, Any]:
    report = build_report(run_dir)
    md = render_markdown(report)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    return report


def print_report_table(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    table = Table(title="LoCoMo per-category results")
    table.add_column("Cat")
    table.add_column("Name")
    table.add_column("N", justify="right")
    table.add_column("Judge Acc", justify="right")
    table.add_column("95% CI")
    table.add_column("F1", justify="right")
    table.add_column("BLEU-1", justify="right")
    for cat, bucket in (metrics.get("by_category") or {}).items():
        table.add_row(
            str(cat),
            CATEGORY_NAMES.get(int(cat), "unknown"),
            str(bucket.get("n", 0)),
            _pct(bucket.get("judge_accuracy")),
            _ci(bucket.get("judge_accuracy_ci95")),
            _num(bucket.get("mean_f1")),
            _num(bucket.get("mean_bleu1")),
        )
    console.print(table)
    console.print(
        f"Headline judge accuracy (excl. cat 5): "
        f"[bold]{_pct(metrics.get('headline_judge_accuracy'))}[/bold] "
        f"{_ci(metrics.get('headline_judge_accuracy_ci95'))}"
    )
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _ci(value: dict[str, float] | None) -> str:
    if not value or value.get("low") is None or value.get("high") is None:
        return "n/a"
    return f"[{_pct(value['low'])}, {_pct(value['high'])}]"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
