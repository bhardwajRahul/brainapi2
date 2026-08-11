from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from recsys.client import BrainAPIClient
from recsys.config import DEFAULT_BRAIN_ID, ML100K_DATASET_PATH, Settings
from recsys.dataset import (
    dataset_stats,
    filter_interactions,
    prepare_ml100k,
)
from recsys.evaluate import (
    ensure_run_dir,
    evaluate_leave_one_out,
    ingest_train_interactions,
    load_interactions,
)
from recsys.mapping import (
    interactions_to_triples,
    leave_one_out_splits,
    structured_ingest_body,
)
from recsys.report import print_report_table, write_report

console = Console()


def _resolve_dataset_path(args: argparse.Namespace, settings: Settings) -> Path:
    if getattr(args, "dataset", None):
        return Path(args.dataset)
    return settings.dataset_path


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    name = (args.name or "ml-100k").strip().lower()
    if name not in {"ml-100k", "movielens-100k", "movielens100k"}:
        console.print(
            f"[red]Unknown dataset[/red] {name!r}. Supported: ml-100k"
        )
        return 1
    out = Path(args.out) if args.out else ML100K_DATASET_PATH
    path = prepare_ml100k(out_path=out, force=args.force, min_rating=args.min_rating)
    rows = load_interactions(path)
    stats = dataset_stats(rows)
    console.print(f"[green]Ready[/green] {path}")
    console.print(stats)
    return 0


def cmd_dataset_stats(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(
            f"[red]Missing[/red] {path}. Run: ./recsys.sh download --name ml-100k"
        )
        return 1
    rows = load_interactions(path)
    filtered = filter_interactions(
        rows,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
    )
    table = Table(title=f"RecSys dataset — {path.name}")
    table.add_column("scope")
    table.add_column("n_interactions")
    table.add_column("n_users")
    table.add_column("n_items")
    table.add_column("ix/user mean")
    for label, subset in (("full", rows), ("filtered", filtered)):
        s = dataset_stats(subset)
        table.add_row(
            label,
            str(s["n_interactions"]),
            str(s["n_users"]),
            str(s["n_items"]),
            f"{s['interactions_per_user_mean']:.1f}",
        )
    console.print(table)
    splits = leave_one_out_splits(filtered)
    console.print(f"[dim]leave-one-out users={len(splits)}[/dim]")
    return 0


def cmd_map(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    rows = load_interactions(path)
    rows = filter_interactions(
        rows,
        min_interactions=getattr(args, "min_interactions", 2) or 2,
        max_users=getattr(args, "max_users", None),
    )
    splits = leave_one_out_splits(rows)
    train = [r for s in splits for r in s["train"]] if args.holdout else rows
    triples = interactions_to_triples(train, include_catalog=not args.no_catalog)
    body = structured_ingest_body(triples, brain_id=settings.brain_id)
    text = json.dumps(body, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {args.out} ({len(triples)} triples)")
    else:
        console.print(text)
    console.print(
        f"[dim]users_with_holdout={len(splits)} interactions={len(rows)} "
        f"mapped={len(train)}[/dim]"
    )
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    rows = load_interactions(path)
    rows = filter_interactions(
        rows,
        min_interactions=2,
        max_users=args.max_users,
    )[: args.limit]
    backend = (getattr(args, "backend", None) or "graph").strip().lower()
    with BrainAPIClient(settings) as client:
        info = ingest_train_interactions(
            client, rows, timeout_s=args.timeout, chunk_size=args.chunk_size
        )
        console.print(info)
        if info["status"] not in {"completed", "partial_failed"}:
            console.print("[red]Smoke ingest failed[/red]")
            return 1
        user_id = str(rows[0]["user_id"])
        if backend == "lightgcn":
            train = client.train_lightgcn(
                epochs=args.epochs, wait=True, timeout_s=args.timeout
            )
            console.print(train.data)
            status = (train.data or {}).get("status")
            nested = ((train.data or {}).get("task") or {}).get("status")
            if status != "completed" and nested != "completed":
                console.print("[red]Smoke train failed[/red]")
                return 1
            rec = client.recommend(user_id, top_k=args.top_k, exclude_seen=False)
            n = len((rec.data or {}).get("items") or [])
            console.print(f"recsys/recommend({user_id}) -> {n} items")
        else:
            rec = client.recommend_graph(
                user_id,
                top_k=args.top_k,
                exclude_seen=False,
                include_attribute_pref=True,
            )
            n = len((rec.data or {}).get("recommendations") or [])
            console.print(f"retrieve/recommend({user_id}) -> {n} items")
        if n < 1:
            console.print("[red]Smoke recommend returned no items[/red]")
            return 1
    console.print("[green]Smoke passed[/green]")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(
            f"[red]Missing[/red] {path}. For MovieLens: ./recsys.sh download --name ml-100k"
        )
        return 1
    rows = load_interactions(path)
    rows = filter_interactions(
        rows,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if not ks:
        ks = (10, 20)
    backend = (args.backend or "graph").strip().lower()

    console.print(
        f"[cyan]Evaluating[/cyan] brain={settings.brain_id} "
        f"dataset={path.name} users≈{len({r['user_id'] for r in rows})} "
        f"run={run_id} backend={backend}"
    )
    with BrainAPIClient(settings) as client:
        result = evaluate_leave_one_out(
            client,
            rows,
            ks=ks,
            timeout_s=args.timeout,
            epochs=args.epochs,
            ingest_chunk_size=args.chunk_size,
            dataset_name=path.name,
            backend=backend,
            include_attribute_pref=not args.no_attribute_pref,
        )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    run_dir = settings.runs_dir / args.run
    eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        console.print(f"[red]Missing[/red] {eval_path}")
        return 1
    eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
    report = write_report(run_dir, eval_result)
    print_report_table(report)
    return 0 if report.get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        description=(
            "Held-out next-item eval on demorecsys: structured ingest → "
            "GET /retrieve/recommend (graph, default) or LightGCN plugin"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv path (default benchmarks/.env)",
    )
    parser.add_argument(
        "--brain",
        default=None,
        help=f"Brain id (default {DEFAULT_BRAIN_ID}; never LoCoMo/BEAM brains)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download MovieLens → JSONL")
    p_dl.add_argument("--name", default="ml-100k", help="Dataset id (ml-100k)")
    p_dl.add_argument("--out", type=str, default=None)
    p_dl.add_argument("--force", action="store_true")
    p_dl.add_argument(
        "--min-rating",
        type=float,
        default=1.0,
        help="Keep ratings >= this (default 1 = all as implicit positives)",
    )
    p_dl.set_defaults(func=cmd_download)

    p_stats = sub.add_parser("dataset-stats", help="Summarize interaction JSONL")
    p_stats.add_argument("--dataset", type=str, default=None)
    p_stats.add_argument("--min-interactions", type=int, default=5)
    p_stats.add_argument("--max-users", type=int, default=100)
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_map = sub.add_parser("map", help="Map interactions JSONL → structured body")
    p_map.add_argument("--dataset", type=str, default=None)
    p_map.add_argument("--out", type=str, default=None)
    p_map.add_argument(
        "--holdout",
        action="store_true",
        help="Map train split only (drop last interaction per user)",
    )
    p_map.add_argument("--no-catalog", action="store_true")
    p_map.add_argument("--min-interactions", type=int, default=2)
    p_map.add_argument("--max-users", type=int, default=None)
    p_map.set_defaults(func=cmd_map)

    p_smoke = sub.add_parser(
        "smoke", help="Ingest → graph or LightGCN recommend smoke"
    )
    p_smoke.add_argument("--dataset", type=str, default=None)
    p_smoke.add_argument("--limit", type=int, default=6)
    p_smoke.add_argument("--max-users", type=int, default=None)
    p_smoke.add_argument("--top-k", type=int, default=10)
    p_smoke.add_argument("--epochs", type=int, default=10)
    p_smoke.add_argument("--chunk-size", type=int, default=200)
    p_smoke.add_argument("--timeout", type=float, default=300.0)
    p_smoke.add_argument(
        "--backend",
        choices=["graph", "lightgcn"],
        default="graph",
        help="Recommend backend (default: graph, no train)",
    )
    p_smoke.set_defaults(func=cmd_smoke)

    p_eval = sub.add_parser(
        "evaluate",
        help="Leave-one-out HitRate/Recall@K (graph recommend or LightGCN)",
    )
    p_eval.add_argument("--dataset", type=str, default=None)
    p_eval.add_argument("--run", type=str, default=None)
    p_eval.add_argument("--ks", type=str, default="10,20")
    p_eval.add_argument("--epochs", type=int, default=20)
    p_eval.add_argument(
        "--max-users",
        type=int,
        default=100,
        help="Cap users for structured-ingest cost (MovieLens tip: start at 50–100)",
    )
    p_eval.add_argument("--min-interactions", type=int, default=5)
    p_eval.add_argument("--chunk-size", type=int, default=200)
    p_eval.add_argument("--timeout", type=float, default=3600.0)
    p_eval.add_argument(
        "--backend",
        choices=["graph", "lightgcn"],
        default="graph",
        help="Recommend backend (default: graph — skips /recsys/train)",
    )
    p_eval.add_argument(
        "--no-attribute-pref",
        action="store_true",
        help="Disable attribute_pref channel on graph backend",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Rebuild report from eval.json")
    p_report.add_argument("--run", type=str, required=True)
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    if args.brain:
        settings.brain_id = args.brain.strip()
        if settings.brain_id.startswith(("beam1m", "locomoconv")):
            raise SystemExit(
                f"Refusing brain_id={settings.brain_id!r}. Use demorecsys."
            )
    return int(args.func(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
