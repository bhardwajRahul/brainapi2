"""MovieLens download + normalize to recsys interaction JSONL."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from recsys.config import DATA_DIR

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_JSONL = DATA_DIR / "movielens_100k.jsonl"
ML100K_CACHE_DIR = DATA_DIR / "movielens" / "ml-100k"

GENRE_NAMES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def download_ml100k(
    *,
    url: str = ML100K_URL,
    cache_dir: Path = ML100K_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download and extract ml-100k; returns directory containing u.data."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_file = cache_dir / "u.data"
    if data_file.exists() and not force:
        return cache_dir

    zip_path = cache_dir.parent / "ml-100k.zip"
    if force or not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # archive paths like ml-100k/u.data
            name = Path(member).name
            if name in {"u.data", "u.item", "u.user", "README"}:
                target = cache_dir / name
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
    if not data_file.exists():
        raise FileNotFoundError(f"u.data missing after extract under {cache_dir}")
    return cache_dir


def _load_item_meta(item_path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    if not item_path.exists():
        return meta
    # Latin-1: GroupLens classic files are not UTF-8
    for line in item_path.read_text(encoding="latin-1").splitlines():
        parts = line.split("|")
        if len(parts) < 24:
            continue
        item_id = parts[0].strip()
        title = parts[1].strip()
        flags = parts[5:24]
        genres = [
            GENRE_NAMES[i]
            for i, flag in enumerate(flags)
            if flag == "1" and i < len(GENRE_NAMES) and GENRE_NAMES[i] != "unknown"
        ]
        meta[item_id] = {
            "name": title or f"movie-{item_id}",
            "category": genres[0] if genres else "unknown",
        }
    return meta


def ratings_to_interactions(
    cache_dir: Path,
    *,
    min_rating: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Convert u.data → interaction dicts.

    All ratings >= min_rating become positive implicit edges (behavior=purchase),
    matching standard LightGCN implicit-feedback setups.
    """
    item_meta = _load_item_meta(cache_dir / "u.item")
    rows: list[dict[str, Any]] = []
    for line in (cache_dir / "u.data").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        user_id, item_id, rating_s, ts_s = parts[0], parts[1], parts[2], parts[3]
        try:
            rating = float(rating_s)
            ts = int(ts_s)
        except ValueError:
            continue
        if rating < min_rating:
            continue
        meta = item_meta.get(item_id, {})
        rows.append(
            {
                "user_id": f"mlu{user_id}",
                "item_id": f"mlm{item_id}",
                "behavior": "purchase",
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "rating": rating,
                "category": meta.get("category", "unknown"),
                "title": meta.get("name"),
            }
        )
    rows.sort(key=lambda r: (r["user_id"], r["timestamp"], r["item_id"]))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def prepare_ml100k(
    *,
    out_path: Path = ML100K_JSONL,
    force: bool = False,
    min_rating: float = 1.0,
) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path
    cache = download_ml100k(force=force)
    rows = ratings_to_interactions(cache, min_rating=min_rating)
    return write_jsonl(rows, out_path)


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    users = {str(r["user_id"]) for r in rows}
    items = {str(r["item_id"]) for r in rows}
    per_user = Counter(str(r["user_id"]) for r in rows)
    counts = list(per_user.values())
    return {
        "n_interactions": len(rows),
        "n_users": len(users),
        "n_items": len(items),
        "interactions_per_user_min": min(counts) if counts else 0,
        "interactions_per_user_max": max(counts) if counts else 0,
        "interactions_per_user_mean": (
            sum(counts) / len(counts) if counts else 0.0
        ),
    }


def filter_interactions(
    rows: list[dict[str, Any]],
    *,
    min_interactions: int = 5,
    max_users: int | None = None,
) -> list[dict[str, Any]]:
    """Keep users with ≥ min_interactions; optionally take first max_users by id."""
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_user.setdefault(str(row["user_id"]), []).append(row)
    eligible = sorted(
        uid for uid, xs in by_user.items() if len(xs) >= min_interactions
    )
    if max_users is not None:
        eligible = eligible[: max(0, int(max_users))]
    keep = set(eligible)
    out = [r for r in rows if str(r["user_id"]) in keep]
    out.sort(key=lambda r: (str(r["user_id"]), r.get("timestamp") or "", str(r["item_id"])))
    return out
