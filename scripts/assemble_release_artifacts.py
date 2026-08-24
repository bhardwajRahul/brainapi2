#!/usr/bin/env python3
"""Combine light and heavy workflow evidence into one release-gate directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _find(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def _merge_profiles(light: Path, heavy: Path, name: str) -> dict:
    left = json.loads(_find(light, name).read_text(encoding="utf-8"))
    right = json.loads(_find(heavy, name).read_text(encoding="utf-8"))
    if set(left) != {"light"} or set(right) != {"heavy"}:
        raise RuntimeError(f"Unexpected profile keys in {name}")
    return {**left, **right}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--light", type=Path, required=True)
    parser.add_argument("--heavy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("smoke.json", "latency.json"):
        merged = _merge_profiles(args.light, args.heavy, name)
        (output / name).write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for root, names in (
        (
            args.light,
            (
                "restore-light.json",
                "security.json",
                "brainapi.spdx.json",
                "backup-light-manifest.json",
                "brainapi-image.tar.gz",
            ),
        ),
        (args.heavy, ("restore-heavy.json", "backup-heavy-manifest.json")),
    ):
        for name in names:
            shutil.copy2(_find(root, name), output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
