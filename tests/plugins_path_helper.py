"""Load plugin modules without installing the plugin package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "recsys-gnn"


def _load(name: str, path: Path):
    # Ensure plugin root is on path so `from models.X` works inside modules.
    plugin_str = str(PLUGIN)
    if plugin_str not in sys.path:
        sys.path.insert(0, plugin_str)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_export_edges():
    return _load(
        "recsys_gnn_export_edges",
        PLUGIN / "models" / "export_edges.py",
    )
