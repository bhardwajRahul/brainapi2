from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARKS_ROOT / "data"
RUNS_DIR = BENCHMARKS_ROOT / "runs"
DEFAULT_DATASET_PATH = DATA_DIR / "search_toy.jsonl"
DEFAULT_BRAIN_ID = "searchbenchsmoke"
REQUIRED_PREFIX = "searchbench"
FORBIDDEN_BRAINS = frozenset(
    {
        "beam1m1clean",
        "demorecsys",
        "locomoconv26",
        "locomoconv26clean",
        "locomoconv26nostorm",
    }
)
FORBIDDEN_PREFIXES = (
    "beam",
    "demorecsys",
    "lme",
    "locomoconv",
    "longmemeval",
)
FROZEN_STRUCTURED_BRAINS = frozenset(
    {
        "searchbenchwands",
        "searchbenchesci74",
        "searchbenchescies",
        "searchbenchesciltr2",
    }
)
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})


def validate_brain_id(brain_id: str) -> str:
    bid = (brain_id or "").strip()
    if not bid:
        raise SystemExit("brain_id is required.")
    lowered = bid.lower()
    forbidden = {item.lower() for item in FORBIDDEN_BRAINS}
    if lowered in forbidden or lowered.startswith(FORBIDDEN_PREFIXES):
        raise SystemExit(
            f"Refusing brain_id={bid!r}. Use a dedicated searchbench* brain "
            f"(default {DEFAULT_BRAIN_ID}). Never wipe eval memory or recsys brains."
        )
    if not lowered.startswith(REQUIRED_PREFIX):
        raise SystemExit(
            f"Refusing brain_id={bid!r}. Search eval brains must start with "
            f"{REQUIRED_PREFIX!r} (default {DEFAULT_BRAIN_ID})."
        )
    return bid


@dataclass
class Settings:
    brainapi_url: str
    brainpat_token: str
    dataset_path: Path
    runs_dir: Path
    brain_id: str

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            env_file = BENCHMARKS_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        brain_id = validate_brain_id(
            os.getenv("SEARCH_BRAIN_ID", DEFAULT_BRAIN_ID) or DEFAULT_BRAIN_ID
        )
        return cls(
            brainapi_url=os.getenv("BRAINAPI_URL", "http://localhost:8000").rstrip("/"),
            brainpat_token=os.getenv("BRAINPAT_TOKEN", ""),
            dataset_path=Path(
                os.getenv("SEARCH_DATASET_PATH", str(DEFAULT_DATASET_PATH))
            ),
            runs_dir=Path(os.getenv("BENCH_RUNS_DIR", str(RUNS_DIR))),
            brain_id=brain_id,
        )

    def require_brainapi(self) -> None:
        if not self.brainpat_token:
            raise SystemExit(
                "BRAINPAT_TOKEN is required. Copy benchmarks/.env.example to "
                "benchmarks/.env and set it."
            )
