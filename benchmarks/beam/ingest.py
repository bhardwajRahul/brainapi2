from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

from beam.client import BrainAPIClient
from beam.config import Settings
from beam.dataset import brain_id_for, iter_ingest_units

console = Console()
_write_lock = threading.Lock()

# Soft cap: higher values queue work but historically crash uvicorn / escalate Neo4j.
RECOMMENDED_MAX_INGEST_CONCURRENCY = 4

# Embed / input size rejects that will not succeed on retry with the same model limits.
_PERMANENT_ERROR_PATTERNS = (
    re.compile(r"maximum context length is 8192", re.IGNORECASE),
    re.compile(r"invalid ['\"]input['\"].*8192", re.IGNORECASE),
)

_TERMINAL_OK_STATUSES = frozenset({"completed", "partial_failed", "dry_run"})
_TERMINAL_SKIP_STATUSES = frozenset(
    {"completed", "partial_failed", "dry_run", "permanent_failed"}
)


@dataclass
class IngestRecord:
    sample_id: str
    brain_id: str
    unit_id: str
    session_key: str
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


def is_permanent_ingest_error(error: str | None) -> bool:
    """True for hard failures that must not be retried on resume (e.g. embed 8192)."""
    if not error:
        return False
    return any(pat.search(error) for pat in _PERMANENT_ERROR_PATTERNS)


def normalize_ingest_status(status: str | None, error: str | None = None) -> str:
    """Map API/harness statuses; upgrade permanent embed fails to permanent_failed."""
    raw = (status or "unknown").strip() or "unknown"
    if raw == "permanent_failed":
        return raw
    if raw == "failed" and is_permanent_ingest_error(error):
        return "permanent_failed"
    if is_permanent_ingest_error(error) and raw not in _TERMINAL_OK_STATUSES:
        return "permanent_failed"
    return raw


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
    """Unit keys that must not be re-ingested on resume.

    Includes successful terminals plus permanent_failed (and legacy `failed` rows
    whose error matches the embed-8192 permanent pattern).
    """
    last: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(ingest_path):
        key = f"{row.get('sample_id')}::{row.get('unit_id')}"
        last[key] = row

    done: set[str] = set()
    for key, row in last.items():
        status = normalize_ingest_status(row.get("status"), row.get("error"))
        if status in _TERMINAL_SKIP_STATUSES:
            done.add(key)
    return done


def ingest_samples(
    settings: Settings,
    samples: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    limit_turns: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    task_timeout_s: float = 900.0,
    brain_override: str | None = None,
) -> list[IngestRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ingest_path = run_dir / "ingest.jsonl"
    already_done = completed_unit_ids(ingest_path) if resume else set()

    if brain_override and len(samples) > 1:
        raise SystemExit("--brain can only be used with a single --sample")

    workers = max(1, int(concurrency))
    if workers > RECOMMENDED_MAX_INGEST_CONCURRENCY:
        console.print(
            f"[yellow]Warning: concurrency={workers} exceeds recommended "
            f"max {RECOMMENDED_MAX_INGEST_CONCURRENCY} for BEAM "
            f"(uvicorn / Neo4j contention). Prefer 2–4.[/yellow]"
        )

    jobs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        size = str(sample["size"])
        conversation_id = str(sample["conversation_id"])
        brain_id = brain_override or brain_id_for(size, conversation_id)
        for unit in iter_ingest_units(sample, limit_turns=limit_turns):
            key = f"{sample_id}::{unit['unit_id']}"
            if key in already_done:
                continue
            jobs.append(
                {
                    "sample_id": sample_id,
                    "brain_id": brain_id,
                    "unit": unit,
                }
            )

    if not jobs:
        console.print("[green]Nothing to ingest (all units already completed).[/green]")
        return []

    console.print(
        f"Ingesting {len(jobs)} turns "
        f"(concurrency={workers}, dry_run={dry_run})"
    )

    records: list[IngestRecord] = []

    def _one(job: dict[str, Any]) -> IngestRecord:
        unit = job["unit"]
        if dry_run:
            record = IngestRecord(
                sample_id=job["sample_id"],
                brain_id=job["brain_id"],
                unit_id=unit["unit_id"],
                session_key=unit["session_key"],
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
                if isinstance(error, dict):
                    error = json.dumps(error, ensure_ascii=False)
                elif error is not None:
                    error = str(error)
                status = normalize_ingest_status(status, error)
                cost = (waited.data or {}).get("cost")
                total_llm_tokens = None
                if isinstance(cost, dict):
                    total_llm_tokens = cost.get("total_llm_tokens")
                record = IngestRecord(
                    sample_id=job["sample_id"],
                    brain_id=job["brain_id"],
                    unit_id=unit["unit_id"],
                    session_key=unit["session_key"],
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
                err = str(exc)
                record = IngestRecord(
                    sample_id=job["sample_id"],
                    brain_id=job["brain_id"],
                    unit_id=unit["unit_id"],
                    session_key=unit["session_key"],
                    task_id=None,
                    status=normalize_ingest_status("failed", err),
                    error=err,
                    submit_latency_ms=None,
                    wait_latency_ms=None,
                    dry_run=False,
                    timestamp=_utc_now(),
                )
            append_jsonl(ingest_path, asdict(record))
            return record

    with Progress(console=console) as progress:
        task = progress.add_task("ingest", total=len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result())
                progress.advance(task)

    ok = sum(1 for r in records if r.status in {"completed", "dry_run"})
    partial = sum(1 for r in records if r.status == "partial_failed")
    permanent = sum(1 for r in records if r.status == "permanent_failed")
    failed = sum(1 for r in records if r.status == "failed")
    console.print(
        f"Ingest done: ok={ok} partial_failed={partial} "
        f"permanent_failed={permanent} failed={failed} "
        f"-> {ingest_path}"
    )
    return records


def ensure_run_dir(settings: Settings, run_id: str | None = None) -> tuple[str, Path]:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("beam-%Y%m%dT%H%M%SZ")
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
