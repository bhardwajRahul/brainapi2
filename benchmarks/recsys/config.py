from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARKS_ROOT / "data"
RUNS_DIR = BENCHMARKS_ROOT / "runs"
DEFAULT_DATASET_PATH = DATA_DIR / "recsys_toy.jsonl"
ML100K_DATASET_PATH = DATA_DIR / "movielens_100k.jsonl"
DEFAULT_BRAIN_ID = "demorecsys"
FORBIDDEN_BRAINS = frozenset(
    {
        "beam1m1clean",
        "locomoconv26",
        "locomoconv26clean",
        "locomoconv26nostorm",
    }
)
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})


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

        brain_id = (
            os.getenv("RECSYS_BRAIN_ID", DEFAULT_BRAIN_ID) or DEFAULT_BRAIN_ID
        ).strip()
        if brain_id in FORBIDDEN_BRAINS or brain_id.startswith(
            ("beam1m", "locomoconv")
        ):
            raise SystemExit(
                f"Refusing brain_id={brain_id!r}. Use a dedicated recsys brain "
                f"(default {DEFAULT_BRAIN_ID}). Never wipe eval memory brains."
            )

        return cls(
            brainapi_url=os.getenv("BRAINAPI_URL", "http://localhost:8000").rstrip("/"),
            brainpat_token=os.getenv("BRAINPAT_TOKEN", ""),
            dataset_path=Path(
                os.getenv("RECSYS_DATASET_PATH", str(DEFAULT_DATASET_PATH))
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
