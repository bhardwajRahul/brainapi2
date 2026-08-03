from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from beam.config import BENCHMARKS_ROOT, Settings, model_family
from beam.prompts import (
    ANSWER_SYSTEM,
    JUDGE_PROMPT_VARIANT,
    UNIFIED_LLM_JUDGE_BASE_PROMPT,
    prompt_sha256,
)

REPO_ROOT = BENCHMARKS_ROOT.parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_manifest_sha256(data_dir: Path) -> str | None:
    if not data_dir.exists():
        return None
    digest = hashlib.sha256()
    paths = sorted(p for p in data_dir.rglob("*") if p.is_file())
    if not paths:
        return None
    for path in paths:
        rel = str(path.relative_to(data_dir))
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build_provenance(settings: Settings, *, size: str | None = None) -> dict[str, Any]:
    data_root = settings.beam_data_dir
    scoped = data_root / size if size else data_root
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "answer_prompt_sha256": prompt_sha256(ANSWER_SYSTEM),
        "judge_prompt_sha256": prompt_sha256(UNIFIED_LLM_JUDGE_BASE_PROMPT),
        "judge_prompt_variant": JUDGE_PROMPT_VARIANT,
        "dataset_path": str(scoped),
        "dataset_sha256": dir_manifest_sha256(scoped),
        "answer_model": settings.answer_model,
        "answer_model_family": model_family(settings.answer_model),
        "answer_llm_base_url": settings.llm_base_url,
        "judge_model": settings.judge_model,
        "judge_model_family": model_family(settings.judge_model),
        "judge_provider": settings.judge_provider(),
        "judge_shares_answer_family": settings.judge_shares_answer_family,
        "hf_dataset": settings.hf_dataset,
        "bench_profile": settings.bench_profile,
        "sc_samples": settings.sc_samples,
        "sc_temperature": settings.sc_temperature,
        "gap_fill": settings.gap_fill,
    }
