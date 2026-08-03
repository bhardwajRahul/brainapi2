from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

from longmemeval.client import BrainAPIClient
from longmemeval.config import Settings
from longmemeval.dataset import brain_id_for, iter_ingest_units

console = Console()
_write_lock = threading.Lock()


@dataclass
class IngestRecord:
    question_id: str
    brain_id: str
    unit_id: str
    session_id: str
    task_id: str | None
    status: str
    error: str | None
    submit_latency_ms: float | None
    wait_latency_ms: float | None
    dry_run: bool
    timestamp: str
    cost: dict[str, Any] | None = None
    total_llm_tokens: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_unit_ids(ingest_path: Path) -> set[str]:
    done = set()
    for row in load_jsonl(ingest_path):
        status = row.get("status")
        if status in {"completed", "partial_failed", "dry_run"}:
            key = f"{row.get('question_id')}::{row.get('unit_id')}"
            done.add(key)
    return done


def ingest_questions(
    settings: Settings,
    questions: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    limit_sessions: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    task_timeout_s: float = 900.0,
) -> list[IngestRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ingest_path = run_dir / "ingest.jsonl"
    already_done = completed_unit_ids(ingest_path) if resume else set()

    jobs: list[dict[str, Any]] = []
    for entry in questions:
        question_id = str(entry["question_id"])
        brain_id = brain_id_for(question_id)
        for unit in iter_ingest_units(entry, limit_sessions=limit_sessions):
            key = f"{question_id}::{unit['unit_id']}"
            if key in already_done:
                continue
            jobs.append(
                {
                    "question_id": question_id,
                    "brain_id": brain_id,
                    "unit": unit,
                }
            )

    if not jobs:
        console.print("[green]Nothing to ingest (all units already completed).[/green]")
        return []

    console.print(
        f"Ingesting {len(jobs)} units "
        f"(concurrency={concurrency}, dry_run={dry_run})"
    )

    records: list[IngestRecord] = []

    def _one(job: dict[str, Any]) -> IngestRecord:
        unit = job["unit"]
        if dry_run:
            record = IngestRecord(
                question_id=job["question_id"],
                brain_id=job["brain_id"],
                unit_id=unit["unit_id"],
                session_id=unit["session_id"],
                task_id=None,
                status="dry_run",
                error=None,
                submit_latency_ms=None,
                wait_latency_ms=None,
                dry_run=True,
                timestamp=_utc_now(),
            )
            append_jsonl(ingest_path, asdict(record))
            return record

        with BrainAPIClient(settings) as client:
            try:
                submitted = client.ingest_text(
                    unit["text"],
                    job["brain_id"],
                    source_timestamp=unit.get("source_timestamp"),
                )
                task_id = (submitted.data or {}).get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in ingest response: {submitted.data}")
                waited = client.wait_for_task(
                    task_id, job["brain_id"], timeout_s=task_timeout_s
                )
                status = (waited.data or {}).get("status", "unknown")
                error = (waited.data or {}).get("error")
                cost = (waited.data or {}).get("cost")
                total_llm_tokens = None
                if isinstance(cost, dict):
                    total_llm_tokens = cost.get("total_llm_tokens")
                record = IngestRecord(
                    question_id=job["question_id"],
                    brain_id=job["brain_id"],
                    unit_id=unit["unit_id"],
                    session_id=unit["session_id"],
                    task_id=task_id,
                    status=status,
                    error=error,
                    submit_latency_ms=submitted.latency_ms,
                    wait_latency_ms=waited.latency_ms,
                    dry_run=False,
                    timestamp=_utc_now(),
                    cost=cost if isinstance(cost, dict) else None,
                    total_llm_tokens=(
                        int(total_llm_tokens) if total_llm_tokens is not None else None
                    ),
                )
            except Exception as exc:
                record = IngestRecord(
                    question_id=job["question_id"],
                    brain_id=job["brain_id"],
                    unit_id=unit["unit_id"],
                    session_id=unit["session_id"],
                    task_id=None,
                    status="failed",
                    error=str(exc),
                    submit_latency_ms=None,
                    wait_latency_ms=None,
                    dry_run=False,
                    timestamp=_utc_now(),
                )
            append_jsonl(ingest_path, asdict(record))
            return record

    with Progress(console=console) as progress:
        task = progress.add_task("ingest", total=len(jobs))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(_one, job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result())
                progress.advance(task)

    ok = sum(1 for r in records if r.status in {"completed", "dry_run"})
    partial = sum(1 for r in records if r.status == "partial_failed")
    failed = sum(1 for r in records if r.status == "failed")
    console.print(
        f"Ingest done: ok={ok} partial_failed={partial} failed={failed} "
        f"-> {ingest_path}"
    )
    return records


def ensure_run_dir(settings: Settings, run_id: str | None = None) -> tuple[str, Path]:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(manifest)
    existing["updated_at"] = _utc_now()
    if "created_at" not in existing:
        existing["created_at"] = _utc_now()
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
