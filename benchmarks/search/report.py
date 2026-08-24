from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from search.config import BENCHMARKS_ROOT

console = Console()

REPORTS_PATH = BENCHMARKS_ROOT / "REPORTS.json"
_BENCHMARK_ID = "search"
_BENCHMARK_NAME = "Search (hybrid BM25 + dense)"
_REPORTS_DESCRIPTION = (
    "BrainAPI benchmark results. Top published scores across suites. "
    "Updated when a suite evaluate/report completes successfully."
)


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BENCHMARKS_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def build_report(run_dir: Path, eval_result: dict[str, Any]) -> dict[str, Any]:
    metrics = eval_result.get("metrics") or {}
    status = str(eval_result.get("status") or "failed")
    n_queries = int(eval_result.get("n_queries") or 0)
    if status != "ok":
        status = "failed"
    if n_queries < 1:
        status = "failed"
    if (eval_result.get("ingest") or {}).get("status") not in {
        "completed",
        "partial_failed",
    }:
        status = "failed"

    report = {
        "suite": _BENCHMARK_ID,
        "status": status,
        "run_id": run_dir.name,
        "brain_id": eval_result.get("brain_id"),
        "dataset": eval_result.get("dataset"),
        "fusion": eval_result.get("fusion"),
        "rerank": eval_result.get("rerank") or "none",
        "channels": eval_result.get("channels") or ["passages"],
        "rank_pool": bool(eval_result.get("rank_pool")),
        "rank_pool_ce": bool(eval_result.get("rank_pool_ce")),
        "ce_model": eval_result.get("ce_model"),
        "skip_enrichment": eval_result.get("skip_enrichment"),
        "n_queries": n_queries,
        "n_docs": eval_result.get("n_docs"),
        "k": eval_result.get("k"),
        "ndcg@10": metrics.get("ndcg@10"),
        "ndcg@20": metrics.get("ndcg@20"),
        "ndcg@50": metrics.get("ndcg@50"),
        "ndcg@100": metrics.get("ndcg@100"),
        "ndcg": metrics.get("ndcg"),
        "recall@10": metrics.get("recall@10"),
        "recall@20": metrics.get("recall@20"),
        "recall@50": metrics.get("recall@50"),
        "recall@100": metrics.get("recall@100"),
        "mrr": metrics.get("mrr"),
        "pool_coverage": metrics.get("pool_coverage"),
        "p50_retrieve_ms": metrics.get("p50_retrieve_ms"),
        "p95_retrieve_ms": metrics.get("p95_retrieve_ms"),
        "metrics": metrics,
        "ingest": eval_result.get("ingest"),
        "git_sha": _git_sha(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "protocol": eval_result.get("protocol")
        or (
            "labeled toy search: POST /ingest/ → POST /retrieve/search "
            f"(fusion={eval_result.get('fusion') or 'rrf'}, "
            f"rerank={eval_result.get('rerank') or 'none'}, "
            f"rank_pool={bool(eval_result.get('rank_pool'))})"
        ),
    }
    return report


def write_report(run_dir: Path, eval_result: dict[str, Any]) -> dict[str, Any]:
    report = build_report(run_dir, eval_result)
    (run_dir / "eval.json").write_text(
        json.dumps(eval_result, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = render_markdown(report)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    if report.get("status") == "ok":
        update_reports_json(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Search report — `{report.get('run_id')}`",
        "",
        f"- status: **{report.get('status')}**",
        f"- dataset: `{report.get('dataset')}`",
        f"- brain_id: `{report.get('brain_id')}`",
        f"- fusion: `{report.get('fusion')}`",
        f"- rerank: `{report.get('rerank')}`",
        f"- channels: `{','.join(report.get('channels') or ['passages'])}`",
        f"- rank_pool: `{report.get('rank_pool')}`",
        f"- skip_enrichment: `{report.get('skip_enrichment')}`",
        f"- k: {report.get('k')}",
        f"- n_queries: {report.get('n_queries')}",
        f"- nDCG@10: {report.get('ndcg@10')}",
        f"- nDCG@20: {report.get('ndcg@20')}",
        f"- nDCG@50: {report.get('ndcg@50')}",
        f"- nDCG (full list): {report.get('ndcg')}",
        f"- Recall@10: {report.get('recall@10')}",
        f"- Recall@20: {report.get('recall@20')}",
        f"- Recall@50: {report.get('recall@50')}",
        f"- pool_coverage: {report.get('pool_coverage')}",
        f"- MRR: {report.get('mrr')}",
        f"- p50 retrieve ms (ex-embed): {report.get('p50_retrieve_ms')}",
        f"- p95 retrieve ms (ex-embed): {report.get('p95_retrieve_ms')}",
        "",
        "Never mutate LoCoMo / BEAM / LongMemEval / RecSys brains or their ledger rows.",
        "",
    ]
    return "\n".join(lines)


def print_report_table(report: dict[str, Any]) -> None:
    table = Table(title=f"Search {report.get('run_id')}")
    table.add_column("metric")
    table.add_column("value")
    for key in (
        "status",
        "dataset",
        "brain_id",
        "fusion",
        "rerank",
        "channels",
        "rank_pool",
        "skip_enrichment",
        "n_queries",
        "k",
        "ndcg@10",
        "ndcg@20",
        "ndcg@50",
        "ndcg",
        "recall@10",
        "recall@20",
        "recall@50",
        "pool_coverage",
        "mrr",
        "p50_retrieve_ms",
        "p95_retrieve_ms",
    ):
        table.add_row(key, str(report.get(key)))
    console.print(table)


def entry_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    if report.get("status") != "ok":
        return None
    n_queries = int(report.get("n_queries") or 0)
    if n_queries < 1:
        return None
    entry = {
        "run_id": report.get("run_id"),
        "brain": report.get("brain_id"),
        "dataset": report.get("dataset"),
        "fusion": report.get("fusion"),
        "rerank": report.get("rerank") or "none",
        "channels": report.get("channels") or ["passages"],
        "rank_pool": bool(report.get("rank_pool")),
        "rank_pool_ce": bool(report.get("rank_pool_ce")),
        "n_queries": n_queries,
        "k": report.get("k"),
        "ndcg@10": report.get("ndcg@10"),
        "ndcg@20": report.get("ndcg@20"),
        "ndcg@50": report.get("ndcg@50"),
        "ndcg": report.get("ndcg"),
        "recall@10": report.get("recall@10"),
        "recall@20": report.get("recall@20"),
        "recall@50": report.get("recall@50"),
        "mrr": report.get("mrr"),
        "pool_coverage": report.get("pool_coverage"),
        "p50_retrieve_ms": report.get("p50_retrieve_ms"),
        "p95_retrieve_ms": report.get("p95_retrieve_ms"),
        "git_sha": report.get("git_sha"),
        "report_path": f"runs/{report.get('run_id')}/report.json",
        "recorded_at": report.get("recorded_at"),
        "protocol": report.get("protocol"),
    }
    if report.get("brain_id") == "searchbenchwandsgraph":
        entry["claim"] = "architecture-demo"
    if report.get("brain_id") == "searchbenchjdslice":
        return None
    return entry


def _empty_reports_ledger() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "description": _REPORTS_DESCRIPTION,
        "benchmarks": {},
    }


def update_reports_json(report: dict[str, Any]) -> None:
    entry = entry_from_report(report)
    if entry is None:
        return

    if REPORTS_PATH.exists():
        data = json.loads(REPORTS_PATH.read_text(encoding="utf-8"))
    else:
        data = _empty_reports_ledger()

    suite = data.setdefault("benchmarks", {}).setdefault(
        _BENCHMARK_ID,
        {"name": _BENCHMARK_NAME, "leaderboard": []},
    )
    leaderboard = list(suite.get("leaderboard") or [])
    run_id = entry["run_id"]
    lb_idx = next(
        (i for i, row in enumerate(leaderboard) if row.get("run_id") == run_id),
        None,
    )
    if lb_idx is not None:
        leaderboard[lb_idx] = entry
    else:
        leaderboard.append(entry)

    suite["name"] = _BENCHMARK_NAME
    suite["leaderboard"] = leaderboard
    data["benchmarks"][_BENCHMARK_ID] = suite
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["description"] = data.get("description") or _REPORTS_DESCRIPTION

    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="REPORTS.",
        suffix=".json.tmp",
        dir=str(REPORTS_PATH.parent),
    )
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        Path(tmp_name).replace(REPORTS_PATH)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
