from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from recsys.config import BENCHMARKS_ROOT

console = Console()

REPORTS_PATH = BENCHMARKS_ROOT / "REPORTS.json"
_BENCHMARK_ID = "recsys"
_BENCHMARK_NAME = "RecSys (graph recommend / LightGCN)"
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
    status = "ok"
    if (eval_result.get("ingest") or {}).get("status") not in {
        "completed",
        "partial_failed",
    }:
        status = "failed"
    train = eval_result.get("train") or {}
    backend = (eval_result.get("backend") or "").lower()
    model = eval_result.get("model") or "lightgcn"
    train_ok = (
        train.get("status") == "skipped"
        or backend == "graph"
        or model == "graph-recommend"
        or train.get("status") == "completed"
        or ((train.get("task") or {}).get("status") == "completed")
    )
    if not train_ok:
        status = "failed"
    if not eval_result.get("n_users"):
        status = "failed"

    protocol = (
        "held-out next-item: structured ingest → GET /retrieve/recommend "
        "(train-free graph)"
        if backend == "graph" or model == "graph-recommend"
        else (
            "held-out next-item: structured ingest → POST /recsys/train "
            "(LightGCN) → GET /recsys/recommend"
        )
    )

    report = {
        "suite": _BENCHMARK_ID,
        "status": status,
        "run_id": run_dir.name,
        "brain_id": eval_result.get("brain_id"),
        "model": model,
        "backend": eval_result.get("backend"),
        "dataset": eval_result.get("dataset"),
        "n_users": eval_result.get("n_users"),
        "ks": eval_result.get("ks"),
        "hit_rate@10": metrics.get("hit_rate@10"),
        "hit_rate@20": metrics.get("hit_rate@20"),
        "recall@10": metrics.get("recall@10"),
        "recall@20": metrics.get("recall@20"),
        "popularity_hit_rate@10": metrics.get("popularity_hit_rate@10"),
        "popularity_hit_rate@20": metrics.get("popularity_hit_rate@20"),
        "metrics": metrics,
        "ingest": eval_result.get("ingest"),
        "train": train,
        "git_sha": _git_sha(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
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
        f"# RecSys report — `{report.get('run_id')}`",
        "",
        f"- status: **{report.get('status')}**",
        f"- model: `{report.get('model')}`",
        f"- dataset: `{report.get('dataset')}`",
        f"- brain_id: `{report.get('brain_id')}`",
        f"- n_users: {report.get('n_users')}",
        f"- HitRate@10: {report.get('hit_rate@10')}",
        f"- HitRate@20: {report.get('hit_rate@20')}",
        f"- Recall@10: {report.get('recall@10')}",
        f"- Recall@20: {report.get('recall@20')}",
        f"- popularity HitRate@10: {report.get('popularity_hit_rate@10')}",
        f"- popularity HitRate@20: {report.get('popularity_hit_rate@20')}",
        "",
        "Never mutate LoCoMo / BEAM / LongMemEval brains or their ledger rows.",
        "",
    ]
    return "\n".join(lines)


def print_report_table(report: dict[str, Any]) -> None:
    table = Table(title=f"RecSys {report.get('run_id')}")
    table.add_column("metric")
    table.add_column("value")
    for key in (
        "status",
        "model",
        "dataset",
        "brain_id",
        "n_users",
        "hit_rate@10",
        "hit_rate@20",
        "recall@10",
        "recall@20",
        "popularity_hit_rate@10",
        "popularity_hit_rate@20",
    ):
        table.add_row(key, str(report.get(key)))
    console.print(table)


def entry_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    if report.get("status") != "ok":
        return None
    n_users = int(report.get("n_users") or 0)
    if n_users < 1:
        return None
    return {
        "run_id": report.get("run_id"),
        "brain": report.get("brain_id"),
        "model": report.get("model") or "lightgcn",
        "dataset": report.get("dataset"),
        "n_users": n_users,
        "hit_rate@10": report.get("hit_rate@10"),
        "hit_rate@20": report.get("hit_rate@20"),
        "recall@10": report.get("recall@10"),
        "recall@20": report.get("recall@20"),
        "popularity_hit_rate@10": report.get("popularity_hit_rate@10"),
        "popularity_hit_rate@20": report.get("popularity_hit_rate@20"),
        "git_sha": report.get("git_sha"),
        "report_path": f"runs/{report.get('run_id')}/report.json",
        "recorded_at": report.get("recorded_at"),
    }


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
