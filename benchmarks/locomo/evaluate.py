from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress

from locomo.answer import answer_question
from locomo.client import BrainAPIClient
from locomo.config import Settings
from locomo.dataset import brain_id_for
from locomo.ingest import append_jsonl, load_jsonl
from locomo.judge import judge_answer
from locomo.metrics import bleu1_score, f1_score

console = Console()
_write_lock = threading.Lock()


@dataclass
class AnswerRecord:
    sample_id: str
    brain_id: str
    qa_index: int
    question: str
    gold: str
    prediction: str
    category: int
    evidence: list[str]
    judge_correct: bool
    judge_reason: str
    judge_raw: str
    f1: float
    bleu1: float
    retrieve_latency_ms: float | None
    answer_latency_ms: float | None
    judge_latency_ms: float | None
    answer_model: str | None
    judge_model: str | None
    answer_total_tokens: int | None
    judge_total_tokens: int | None
    text_context: str
    error: str | None
    timestamp: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_key(sample_id: str, qa_index: int) -> str:
    return f"{sample_id}::{qa_index}"


def completed_qa_keys(answers_path: Path) -> set[str]:
    done = set()
    for row in load_jsonl(answers_path):
        if row.get("error"):
            continue
        if row.get("prediction") is None:
            continue
        done.add(qa_key(str(row.get("sample_id")), int(row.get("qa_index"))))
    return done


def iter_qa_jobs(
    samples: list[dict[str, Any]],
    *,
    categories: set[int] | None = None,
    skip_adversarial: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        brain_id = brain_id_for(sample_id)
        for idx, qa in enumerate(sample.get("qa") or []):
            cat = int(qa.get("category") or 0)
            if skip_adversarial and cat == 5:
                continue
            if categories is not None and cat not in categories:
                continue
            jobs.append(
                {
                    "sample_id": sample_id,
                    "brain_id": brain_id,
                    "qa_index": idx,
                    "qa": qa,
                }
            )
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def evaluate_samples(
    settings: Settings,
    samples: list[dict[str, Any]],
    *,
    run_dir: Path,
    concurrency: int = 2,
    categories: set[int] | None = None,
    skip_adversarial: bool = True,
    limit: int | None = None,
    resume: bool = True,
    historical_limit: int = 10,
) -> list[AnswerRecord]:
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_path = run_dir / "answers.jsonl"
    already_done = completed_qa_keys(answers_path) if resume else set()

    jobs = iter_qa_jobs(
        samples,
        categories=categories,
        skip_adversarial=skip_adversarial,
        limit=limit,
    )
    jobs = [
        j
        for j in jobs
        if qa_key(j["sample_id"], j["qa_index"]) not in already_done
    ]

    if not jobs:
        console.print("[green]Nothing to evaluate (all QA already scored).[/green]")
        return []

    console.print(
        f"Evaluating {len(jobs)} QA items "
        f"(concurrency={concurrency}, skip_adversarial={skip_adversarial})"
    )

    records: list[AnswerRecord] = []

    def _one(job: dict[str, Any]) -> AnswerRecord:
        qa = job["qa"]
        question = str(qa.get("question") or "")
        gold = str(qa.get("answer") if qa.get("answer") is not None else "")
        category = int(qa.get("category") or 0)
        evidence = list(qa.get("evidence") or [])

        try:
            with BrainAPIClient(settings) as client:
                retrieved = client.retrieve_context(
                    question,
                    job["brain_id"],
                    historical_limit=historical_limit,
                )
            context = retrieved.data or {}
            answered = answer_question(settings, question, context)
            judged = judge_answer(
                settings, question, gold, answered.answer
            )
            record = AnswerRecord(
                sample_id=job["sample_id"],
                brain_id=job["brain_id"],
                qa_index=job["qa_index"],
                question=question,
                gold=gold,
                prediction=answered.answer,
                category=category,
                evidence=evidence,
                judge_correct=judged.correct,
                judge_reason=judged.reason,
                judge_raw=judged.raw,
                f1=f1_score(answered.answer, gold),
                bleu1=bleu1_score(answered.answer, gold),
                retrieve_latency_ms=retrieved.latency_ms,
                answer_latency_ms=answered.latency_ms,
                judge_latency_ms=judged.latency_ms,
                answer_model=answered.model,
                judge_model=judged.model,
                answer_total_tokens=answered.total_tokens,
                judge_total_tokens=judged.total_tokens,
                text_context=str(context.get("text_context") or "")[:4000],
                error=None,
                timestamp=_utc_now(),
            )
        except Exception as exc:
            record = AnswerRecord(
                sample_id=job["sample_id"],
                brain_id=job["brain_id"],
                qa_index=job["qa_index"],
                question=question,
                gold=gold,
                prediction="",
                category=category,
                evidence=evidence,
                judge_correct=False,
                judge_reason="",
                judge_raw="",
                f1=0.0,
                bleu1=0.0,
                retrieve_latency_ms=None,
                answer_latency_ms=None,
                judge_latency_ms=None,
                answer_model=None,
                judge_model=None,
                answer_total_tokens=None,
                judge_total_tokens=None,
                text_context="",
                error=str(exc),
                timestamp=_utc_now(),
            )

        with _write_lock:
            append_jsonl(answers_path, asdict(record))
        return record

    with Progress(console=console) as progress:
        task = progress.add_task("evaluate", total=len(jobs))
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(_one, job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result())
                progress.advance(task)

    ok = sum(1 for r in records if not r.error)
    failed = sum(1 for r in records if r.error)
    console.print(f"Evaluate done: ok={ok} failed={failed} -> {answers_path}")
    return records
