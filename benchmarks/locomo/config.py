from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARKS_ROOT / "data"
RUNS_DIR = BENCHMARKS_ROOT / "runs"
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_DATASET_PATH = DATA_DIR / "locomo10.json"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"

TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "partial_failed"})
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


@dataclass
class Settings:
    brainapi_url: str
    brainpat_token: str
    openai_api_key: str
    llm_base_url: str | None
    answer_model: str
    judge_model: str
    dataset_path: Path
    dataset_url: str
    runs_dir: Path

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            env_file = BENCHMARKS_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

        return cls(
            brainapi_url=os.getenv("BRAINAPI_URL", "http://localhost:8000").rstrip("/"),
            brainpat_token=os.getenv("BRAINPAT_TOKEN", ""),
            openai_api_key=api_key,
            llm_base_url=os.getenv("BENCH_LLM_BASE_URL", DEFAULT_LLM_BASE_URL) or None,
            answer_model=os.getenv("BENCH_ANSWER_MODEL", DEFAULT_ANSWER_MODEL),
            judge_model=os.getenv("BENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            dataset_path=Path(
                os.getenv("BENCH_DATASET_PATH", str(DEFAULT_DATASET_PATH))
            ),
            dataset_url=os.getenv("BENCH_DATASET_URL", DEFAULT_DATASET_URL),
            runs_dir=Path(os.getenv("BENCH_RUNS_DIR", str(RUNS_DIR))),
        )

    def require_brainapi(self) -> None:
        if not self.brainpat_token:
            raise SystemExit(
                "BRAINPAT_TOKEN is required. Copy .env.example to .env and set it."
            )

    def require_llm(self) -> None:
        if not self.openai_api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY (or OPENAI_API_KEY) is required for answer/judge "
                "LLM calls. Copy .env.example to .env and set it."
            )
