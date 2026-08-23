from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from search.catalog import (
    DOWNLOAD_ALL_CATALOGS,
    SUPPORTED_CATALOGS,
    catalog_jsonl_path,
    catalog_overwrite_blocked,
    prepare_catalog,
)
from search.client import BrainAPIClient, SearchDisabledError
from search.config import BENCHMARKS_ROOT, DATA_DIR, DEFAULT_BRAIN_ID, Settings, validate_brain_id
from search.dataset import dataset_stats, load_records, split_corpus
from search.evaluate import assert_wandsgraph_node_join, ensure_run_dir, evaluate_search
from search.mapping import load_interaction_rows
from search.report import print_report_table, write_report

console = Console()


def _parse_channels(args: argparse.Namespace) -> list[str] | None:
    raw = getattr(args, "channels", None)
    if not raw:
        return None
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return items or None


def _parse_label_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return items or None


def _parse_extras(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extras must be a JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--extras must be a JSON object")
    return {str(key): str(value) for key, value in data.items() if value is not None} or None


def _resolve_dataset_path(args: argparse.Namespace, settings: Settings) -> Path:
    if getattr(args, "dataset", None):
        return Path(args.dataset)
    return settings.dataset_path


def _skip_enrichment(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "enrich", False))


def _eval_kwargs(args: argparse.Namespace) -> dict:
    interactions = None
    path = getattr(args, "interactions", None)
    if path:
        interactions = load_interaction_rows(Path(path))
    return {
        "fusion": args.fusion,
        "fusion_alpha": getattr(args, "fusion_alpha", None),
        "rerank": args.rerank,
        "mode": getattr(args, "mode", None) or "default",
        "channels": _parse_channels(args),
        "node_labels": _parse_label_list(getattr(args, "node_labels", None)),
        "community_labels": _parse_label_list(getattr(args, "community_labels", None)),
        "expand": getattr(args, "expand", None) or "none",
        "skip_enrichment": _skip_enrichment(args),
        "ingest_graph": bool(getattr(args, "ingest_graph", False)),
        "skip_ingest": bool(getattr(args, "skip_ingest", False)),
        "rank_pool": bool(getattr(args, "rank_pool", False)),
        "personalize": bool(getattr(args, "personalize", False)),
        "interactions": interactions,
        "extras": _parse_extras(getattr(args, "extras", None)),
    }


def cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    from search.config import DATA_DIR
    from search.finetune_esci_4class import held_out_query_ids

    names = [item.strip().lower() for item in str(args.name).split(",") if item.strip()]
    if len(names) == 1 and names[0] in {"all", "*"}:
        names = list(DOWNLOAD_ALL_CATALOGS)
    unknown = [name for name in names if name not in SUPPORTED_CATALOGS]
    if unknown:
        console.print(
            f"[red]Unknown dataset[/red] {unknown!r}. "
            f"Supported: {', '.join(SUPPORTED_CATALOGS)}"
        )
        return 1
    holdout: set[str] = set()
    holdout_raw = str(getattr(args, "holdout_dataset", None) or "").strip()
    if holdout_raw:
        holdout_path = Path(holdout_raw)
        if not holdout_path.is_absolute():
            holdout_path = (BENCHMARKS_ROOT / holdout_path).resolve()
        if not holdout_path.exists():
            console.print(f"[red]Missing holdout dataset[/red] {holdout_path}")
            return 1
        holdout = held_out_query_ids(holdout_path)
        console.print(f"[cyan]Holdout[/cyan] {holdout_path.name} n={len(holdout)}")
    protected = {
        (DATA_DIR / "search_esci.jsonl").resolve(),
        (DATA_DIR / "search_esci_74.jsonl").resolve(),
    }
    timeout = 0
    for name in names:
        try:
            default_out = catalog_jsonl_path(name, locale=args.locale)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        out = Path(args.out) if args.out and len(names) == 1 else default_out
        if not out.is_absolute():
            out = (BENCHMARKS_ROOT / out).resolve()
        split = str(args.split or "test")
        if getattr(args, "dry_stats", False):
            if name != "jdsearch":
                console.print("[red]--dry-stats is only implemented for --name jdsearch[/red]")
                return 1
            from search.jdsearch import collect_jdsearch_stats, print_jdsearch_stats

            console.print(f"[cyan]JDsearch stats[/cyan] (no JSONL write)")
            stats = collect_jdsearch_stats()
            print_jdsearch_stats(stats)
            timeout += 1
            continue
        if catalog_overwrite_blocked(out):
            console.print(f"[red]Refusing to overwrite[/red] {out}")
            return 1
        if split == "train" and out.resolve() in protected:
            console.print(
                f"[red]Refusing to overwrite[/red] {out} with split=train. "
                "Pass --out data/search_esci_ltr200.jsonl"
            )
            return 1
        console.print(f"[cyan]Downloading[/cyan] {name} split={split} → {out}")
        path = prepare_catalog(
            name,
            out_path=out,
            force=args.force,
            max_queries=args.max_queries,
            max_docs=args.max_docs,
            candidates_per_query=args.candidates_per_query,
            locale=args.locale,
            split=split,
            holdout_qids=holdout or None,
        )
        rows = load_records(path)
        stats = dataset_stats(rows)
        console.print(f"[green]Ready[/green] {path}")
        console.print({key: value for key, value in stats.items() if key != "doc_ids"})
        timeout += 1
    return 0 if timeout else 1


def cmd_dataset_stats(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    stats = dataset_stats(rows)
    table = Table(title=f"Search dataset — {path.name}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("n_docs", str(stats["n_docs"]))
    table.add_row("n_queries", str(stats["n_queries"]))
    table.add_row("slices", json.dumps(stats["slices"]))
    table.add_row("graded", str(stats.get("graded")))
    console.print(table)
    return 0


def cmd_smoke(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    docs, queries = split_corpus(rows)
    skip_ingest = bool(getattr(args, "skip_ingest", False))
    if skip_ingest and settings.brain_id == "searchbenchwandsgraph":
        queries = queries[: max(1, min(args.limit_queries, len(queries)))]
    else:
        docs = docs[: args.limit]
        queries = queries[: max(1, min(args.limit_queries, len(queries)))]
    limited = [{"type": "doc", **doc} for doc in docs] + [
        {"type": "query", **query} for query in queries
    ]
    console.print(
        f"[cyan]Smoke[/cyan] brain={settings.brain_id} "
        f"docs={len(docs)} queries={len(queries)}"
    )
    with BrainAPIClient(settings) as client:
        try:
            result = evaluate_search(
                client,
                limited,
                ks=(5, 10, 20),
                timeout_s=args.timeout,
                k=args.k,
                dataset_name=path.name,
                **_eval_kwargs(args),
            )
        except SearchDisabledError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        if result.get("search_error"):
            console.print(f"[red]{result['search_error']}[/red]")
            return 1
        if result.get("ingest", {}).get("status") not in {"completed", "partial_failed"}:
            console.print("[red]Smoke ingest failed[/red]")
            console.print(result.get("ingest"))
            return 1
        if int(result.get("n_docs_mapped") or 0) < 1:
            console.print("[red]Smoke could not map DOCID markers to chunks[/red]")
            return 1
        if int(result.get("n_queries") or 0) < 1:
            console.print("[red]Smoke search returned no scored queries[/red]")
            return 1
        try:
            join = assert_wandsgraph_node_join(client, result)
        except SystemExit as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        if not join.get("skipped"):
            console.print(
                f"node_id={join.get('node_id')} neighbors={join.get('status_code')} "
                f"count={join.get('neighbor_count')}"
            )
    first = (result.get("queries") or [{}])[0]
    console.print(
        f"retrieve/search({first.get('qid')}) -> {first.get('n_hits')} hits "
        f"recall@10={((first.get('metrics') or {}).get('recall@10'))}"
    )
    console.print("[green]Smoke passed[/green]")
    return 0


def cmd_backfill_entity_text(args: argparse.Namespace, settings: Settings) -> int:
    from search.backfill_entities import apply_entity_text_backfill, refuse_entity_backfill
    from search.mapping import catalog_entity_backfill_rows

    try:
        brain_id = refuse_entity_backfill(settings.brain_id)
    except SystemExit as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    docs, _ = split_corpus(load_records(path))
    limit = getattr(args, "limit", None)
    if limit:
        docs = docs[: int(limit)]
    rows = catalog_entity_backfill_rows(docs)
    console.print(
        f"[cyan]ENTITY backfill[/cyan] brain={brain_id} docs={len(rows)}"
    )
    import os
    from dotenv import load_dotenv

    for env_path in (
        Path.home() / ".brainapi" / "source" / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    os.environ.setdefault("DATA_DB", "postgresql")
    os.environ.setdefault("VECTOR_DB", "postgresql")
    os.environ.setdefault("GRAPH_DB", "networkx")
    os.environ.setdefault("SEARCH_ENABLED", "true")
    os.environ.setdefault("ENV", "development")
    from src.core.instances import (
        embeddings_adapter,
        graph_adapter,
        vector_store_adapter,
    )

    summary = apply_entity_text_backfill(
        brain_id=brain_id,
        rows=rows,
        graph=graph_adapter,
        embeddings=embeddings_adapter,
        vector_store=vector_store_adapter,
    )
    console.print(summary)
    if int(summary.get("updated") or 0) < 1:
        console.print("[red]No ENTITY nodes updated[/red]")
        return 1
    console.print("[green]ENTITY backfill done[/green]")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip()) or (5, 10, 20)
    console.print(
        f"[cyan]Evaluating[/cyan] brain={settings.brain_id} "
        f"dataset={path.name} run={run_id} fusion={args.fusion or 'rrf'}"
    )
    with BrainAPIClient(settings) as client:
        try:
            result = evaluate_search(
                client,
                rows,
                ks=ks,
                timeout_s=args.timeout,
                k=args.k,
                dataset_name=path.name,
                limit_docs=getattr(args, "limit_docs", None),
                limit_queries=getattr(args, "limit_queries", None),
                **_eval_kwargs(args),
            )
        except SearchDisabledError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    if result.get("search_error"):
        console.print(f"[red]{result['search_error']}[/red]")
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


def cmd_finetune_ce(args: argparse.Namespace, settings: Settings) -> int:
    from search.finetune_esci_ce import finetune

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    out = Path(args.out)
    console.print(
        f"[cyan]Fine-tune CE[/cyan] dataset={path.name} out={out} "
        f"label_mode={args.label_mode} fields={args.fields}"
    )
    meta = finetune(
        jsonl_path=path,
        out_dir=out,
        max_pairs=int(args.max_pairs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        label_mode=str(args.label_mode),
        fields=str(args.fields),
        base=str(args.base),
        lr=float(args.lr),
        max_length=int(getattr(args, "max_length", 256)),
    )
    console.print(json.dumps(meta, indent=2))
    return 0


def cmd_miss_strata(args: argparse.Namespace, settings: Settings) -> int:
    from search.miss_strata import main_classify

    path = _resolve_dataset_path(args, settings)
    eval_path = settings.runs_dir / args.from_run / "eval.json"
    if not eval_path.exists():
        console.print(f"[red]Missing[/red] {eval_path}")
        return 1
    out = Path(args.out) if args.out else settings.runs_dir / args.from_run / "miss_strata.json"
    taxonomy = main_classify(eval_path=eval_path, dataset_path=path, out_path=out)
    table = Table(title=f"Miss strata {args.from_run}")
    table.add_column("field")
    table.add_column("value")
    for key in (
        "n_queries",
        "n_gold",
        "n_in_top10",
        "n_in_11_to_k",
        "n_missed",
        "stratum_counts",
        "n_total_miss",
        "n_rewritable_total_miss",
        "run_query_side",
        "rewritable_qids",
        "total_miss_qids",
    ):
        table.add_row(key, str(taxonomy.get(key)))
    console.print(table)
    console.print(f"[green]Wrote[/green] {out}")
    return 0


def cmd_query_rewrite(args: argparse.Namespace, settings: Settings) -> int:
    from search.miss_strata import main_classify, write_rewritten_jsonl

    path = _resolve_dataset_path(args, settings)
    eval_path = settings.runs_dir / args.from_run / "eval.json"
    if not eval_path.exists():
        console.print(f"[red]Missing[/red] {eval_path}")
        return 1
    taxonomy = main_classify(eval_path=eval_path, dataset_path=path)
    if not taxonomy.get("run_query_side"):
        console.print(
            "[yellow]No rewritable total-miss qids (skip Task 2).[/yellow] "
            f"total_miss={taxonomy.get('total_miss_qids')}"
        )
        return 0
    dest = Path(args.out) if args.out else path.with_name("search_esci_74_qrewrite.jsonl")
    rows = load_records(path)
    write_rewritten_jsonl(rows, taxonomy, dest)
    console.print(
        f"[green]Wrote[/green] {dest} qids={taxonomy.get('rewritable_qids')}"
    )
    return 0


def cmd_spell_normalize(args: argparse.Namespace, settings: Settings) -> int:
    from search.miss_strata import write_spell_jsonl

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    dest = Path(args.out) if args.out else path.with_name("search_esci_es_spell.jsonl")
    rows = load_records(path)
    summary = write_spell_jsonl(rows, dest)
    console.print(
        f"[green]Wrote[/green] {dest} changed_queries={summary.get('n_queries_changed')}"
    )
    return 0


def cmd_finetune_dense(args: argparse.Namespace, settings: Settings) -> int:
    from search.finetune_esci_dense import finetune_dense

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    eval_path = Path(args.from_eval) if args.from_eval else None
    out = Path(args.out)
    console.print(
        f"[cyan]Fine-tune dense ANCE[/cyan] dataset={path.name} out={out} "
        f"(not Reddy 0.857)"
    )
    meta = finetune_dense(
        jsonl_path=path,
        out_dir=out,
        eval_path=eval_path,
        max_pairs=int(args.max_pairs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        fields=str(args.fields),
        base=str(args.base),
        lr=float(args.lr),
        max_length=int(args.max_length),
    )
    console.print(json.dumps(meta, indent=2))
    return 0


def cmd_local_dense(args: argparse.Namespace, settings: Settings) -> int:
    from search.local_dense import run_local_dense

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    model_name = str(args.model)
    protocol = None
    if "bge" in model_name.lower():
        protocol = (
            "local BGE-base dual-encoder over search_esci_74.jsonl; "
            "not MiniLM ANCE; no query: prefix; not Reddy 0.857"
        )
    if "paraphrase-multilingual-minilm" in model_name.lower() or "italian_smoke" in path.name:
        protocol = (
            "local paraphrase-multilingual-MiniLM-L12-v2 over "
            "search_italian_smoke.jsonl; pipeline only (hits returned); "
            "not text-embedding-3-large quality; not Reddy; not ES n=62"
        )
    console.print(
        f"[cyan]Local dense retrieve[/cyan] dataset={path.name} run={run_id} "
        f"model={model_name} brain={settings.brain_id}"
    )
    result = run_local_dense(
        rows,
        model_name=str(args.model),
        dataset_name=path.name,
        k=int(args.k),
        ks=ks,
        brain_id=settings.brain_id,
        protocol=protocol,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_mine_retrieved_lists(args: argparse.Namespace, settings: Settings) -> int:
    from search.mine_retrieved_lists import mine

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    out = Path(args.out)
    console.print(
        f"[cyan]Mine retrieved lists[/cyan] dataset={path.name} out={out} "
        f"(unlabeled=I; hold out test qids)"
    )
    meta = mine(
        jsonl_path=path,
        out_path=out,
        max_queries=int(args.max_queries),
        k=int(args.k),
        seed=int(args.seed),
    )
    console.print(json.dumps(meta, indent=2))
    return 0


def cmd_export_hybrid_lists(args: argparse.Namespace, settings: Settings) -> int:
    from search.export_hybrid_lists import PROTECTED_OUT_NAMES, export_hybrid_lists
    from search.list_overlap import load_eval_run

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    holdout_raw = str(getattr(args, "holdout_dataset", None) or "").strip()
    holdout = Path(holdout_raw) if holdout_raw else DATA_DIR / "search_esci_74.jsonl"
    if not holdout.is_absolute():
        holdout = (BENCHMARKS_ROOT / holdout).resolve()
    if not holdout.exists():
        console.print(f"[red]Missing holdout[/red] {holdout}")
        return 1
    out = Path(args.out)
    if not out.is_absolute():
        out = (BENCHMARKS_ROOT / out).resolve()
    if out.name in PROTECTED_OUT_NAMES:
        console.print(f"[red]Refusing to overwrite[/red] {out}")
        return 1
    run_dir = settings.runs_dir / str(args.from_run)
    if not (run_dir / "eval.json").exists():
        console.print(f"[red]Missing eval[/red] {run_dir / 'eval.json'}")
        return 1
    eval_result = load_eval_run(run_dir)
    console.print(
        f"[cyan]Export hybrid lists[/cyan] from={args.from_run} "
        f"dataset={path.name} holdout={holdout.name} out={out} "
        f"(unlabeled=I; source=hybrid-k50)"
    )
    try:
        meta = export_hybrid_lists(
            eval_result=eval_result,
            dataset_path=path,
            holdout_path=holdout,
            out_path=out,
            k=int(args.k),
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(json.dumps(meta, indent=2))
    return 0


def cmd_finetune_4class(args: argparse.Namespace, settings: Settings) -> int:
    from search.finetune_esci_4class import finetune

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    lists_path = Path(args.from_lists) if args.from_lists else None
    if lists_path is not None and not lists_path.exists():
        console.print(f"[red]Missing[/red] {lists_path}")
        return 1
    out = Path(args.out)
    if not out.is_absolute():
        out = (BENCHMARKS_ROOT / out).resolve()
    base_model = str(args.base)
    base_path = Path(base_model)
    if not base_path.is_absolute():
        maybe_base = (BENCHMARKS_ROOT / base_path).resolve()
        if maybe_base.exists():
            base_path = maybe_base
            base_model = str(base_path)
    if out.exists() and base_path.exists() and out.resolve() == base_path.resolve():
        console.print(f"[red]Refusing to overwrite base checkpoint[/red] {out}")
        return 1
    lists_source = str(getattr(args, "lists_source", None) or "").strip()
    if lists_path is not None:
        source = lists_source or "retrieved-bm25"
    else:
        source = "pool"
    console.print(
        f"[cyan]Fine-tune 4-class CE[/cyan] dataset={path.name} out={out} "
        f"source={source}"
    )
    meta = finetune(
        jsonl_path=path,
        out_dir=out,
        max_pairs=int(args.max_pairs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        base_model=base_model,
        seed=int(args.seed),
        ckpt_every=int(args.ckpt_every),
        use_class_weights=bool(args.class_weights),
        lists_path=lists_path,
        lists_source=source if lists_path is not None else None,
    )
    console.print(json.dumps(meta, indent=2))
    return 0


def cmd_colbert_local(args: argparse.Namespace, settings: Settings) -> int:
    from search.local_colbert import run_local_colbert

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    console.print(
        f"[cyan]Local ColBERT MaxSim[/cyan] dataset={path.name} run={run_id} "
        f"(not fused with passages; not Reddy 0.857)"
    )
    result = run_local_colbert(
        rows,
        dataset_name=path.name,
        k=int(args.k),
        ks=ks,
        brain_id="harness-local-colbert",
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def _existing_path(path: Path) -> Path:
    if path.exists():
        return path
    from search.config import BENCHMARKS_ROOT

    alt = BENCHMARKS_ROOT / path
    return alt if alt.exists() else path


def cmd_rank_corpus(args: argparse.Namespace, settings: Settings) -> int:
    from search.rank_corpus import DEFAULT_MODEL, load_predict, run_exhaustive_ce

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    model_dir = _existing_path(Path(args.model) if args.model else DEFAULT_MODEL)
    if not model_dir.exists():
        console.print(f"[red]Missing model[/red] {model_dir}")
        return 1
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    console.print(
        f"[cyan]Exhaustive 4-class CE[/cyan] dataset={path.name} run={run_id} "
        f"model={model_dir} (protocol exhaustive-catalog; not Reddy 0.857)"
    )
    predict, model_name = load_predict(model_dir, max_length=int(args.max_length))
    result = run_exhaustive_ce(
        rows,
        predict=predict,
        model_name=model_name,
        dataset_name=path.name,
        k=int(args.k),
        ks=ks,
        brain_id="harness-local-exhaustive",
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_list_overlap(args: argparse.Namespace, settings: Settings) -> int:
    from search.list_overlap import load_eval_run, summarize_overlap

    names = [item.strip() for item in str(args.against_runs).split(",") if item.strip()]
    passages = load_eval_run(settings.runs_dir / args.passages_run)
    passages["run_id"] = args.passages_run
    sidecars: dict = {}
    for name in names:
        eval_result = load_eval_run(settings.runs_dir / name)
        eval_result["run_id"] = name
        sidecars[name] = eval_result
    queries = None
    dataset = getattr(args, "dataset", None)
    if dataset:
        path = Path(dataset)
        if not path.exists():
            path = _existing_path(path)
        if path.exists():
            _, queries = split_corpus(load_records(path))
    summary = summarize_overlap(
        passages,
        sidecars,
        k=int(args.k),
        queries=queries,
    )
    run_id, run_dir = ensure_run_dir(settings, args.run)
    out = run_dir / "overlap.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    headline = {
        name: {
            "unique_gold_hits": row["unique_gold_hits"],
            "queries_with_unique": row["queries_with_unique"],
            "n_queries": row["n_queries"],
        }
        for name, row in summary["runs"].items()
    }
    console.print(json.dumps(headline, indent=2))
    console.print(f"[green]Wrote[/green] {out}")
    return 0


def cmd_union_lists(args: argparse.Namespace, settings: Settings) -> int:
    from search.list_overlap import load_eval_run, run_union

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    names = [item.strip() for item in str(args.from_runs).split(",") if item.strip()]
    evals = []
    for name in names:
        eval_result = load_eval_run(settings.runs_dir / name)
        eval_result["run_id"] = name
        evals.append(eval_result)
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    console.print(
        f"[cyan]Harness RRF union[/cyan] from={names} dataset={path.name} "
        f"run={run_id} (not live graph; not Reddy 0.857)"
    )
    result = run_union(
        evals,
        rows,
        dataset_name=path.name,
        k=int(args.k),
        ks=ks,
        brain_id="harness-local-union",
        run_names=names,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_cascade_lists(args: argparse.Namespace, settings: Settings) -> int:
    from search.list_overlap import load_eval_run, run_cascade

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    passages = load_eval_run(settings.runs_dir / args.passages_run)
    passages["run_id"] = args.passages_run
    names = [item.strip() for item in str(args.from_runs).split(",") if item.strip()]
    sidecars = []
    for name in names:
        eval_result = load_eval_run(settings.runs_dir / name)
        eval_result["run_id"] = name
        sidecars.append(eval_result)
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    console.print(
        f"[cyan]Frozen-head cascade[/cyan] passages={args.passages_run} "
        f"sidecars={names} head_k={args.head_k} dataset={path.name} "
        f"run={run_id} (not live graph; not Reddy 0.857)"
    )
    result = run_cascade(
        passages,
        sidecars,
        rows,
        dataset_name=path.name,
        k=int(args.k),
        head_k=int(args.head_k),
        ks=ks,
        brain_id="harness-local-cascade",
        run_names=[args.passages_run, *names],
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_ltr_head(args: argparse.Namespace, settings: Settings) -> int:
    from search.list_overlap import load_eval_run
    from search.ltr_head import HEAD_RANKNET, PAIR_UNLABELED_ZERO, ce_cache_name, run_ltr_head

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        if not path.is_absolute():
            alt = (BENCHMARKS_ROOT / path).resolve()
            if alt.exists():
                path = alt
        if not path.exists():
            console.print(f"[red]Missing[/red] {path}")
            return 1
    passages = load_eval_run(settings.runs_dir / args.from_run)
    passages["run_id"] = args.from_run
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip()) or (5, 10, 20, 50)
    policy = str(getattr(args, "pair_policy", None) or PAIR_UNLABELED_ZERO)
    head = str(getattr(args, "ltr_model", None) or HEAD_RANKNET)
    ce_raw = str(getattr(args, "ce_model", None) or "").strip()
    ce_model = Path(ce_raw) if ce_raw else None
    if ce_model is not None and not ce_model.is_absolute():
        ce_model = (BENCHMARKS_ROOT / ce_model).resolve()
    if ce_model is not None and not ce_model.exists():
        console.print(f"[red]Missing CE model[/red] {ce_model}")
        return 1
    ce_cache = None
    if ce_model is not None:
        ce_cache = settings.runs_dir / args.from_run / ce_cache_name(ce_model)
    train_run = str(getattr(args, "train_from_run", None) or "").strip()
    train_eval = None
    train_rows = None
    train_ce_cache = None
    train_dataset_name = None
    if train_run:
        train_ds = str(getattr(args, "train_dataset", None) or "").strip()
        if not train_ds:
            console.print("[red]--train-dataset is required with --train-from-run[/red]")
            return 1
        train_path = Path(train_ds)
        if not train_path.is_absolute():
            train_path = (BENCHMARKS_ROOT / train_path).resolve()
        if not train_path.exists():
            console.print(f"[red]Missing train dataset[/red] {train_path}")
            return 1
        train_eval = load_eval_run(settings.runs_dir / train_run)
        train_eval["run_id"] = train_run
        train_rows = load_records(train_path)
        train_dataset_name = train_path.name
        if ce_model is not None:
            train_ce_cache = settings.runs_dir / train_run / ce_cache_name(ce_model)
    mode = "apply" if train_run else "CV"
    console.print(
        f"[cyan]LTR head {mode}[/cyan] from={args.from_run} dataset={path.name} "
        f"train={train_run or 'folds'} {train_dataset_name or ''} "
        f"run={run_id} folds={args.folds} pair={policy} head={head} "
        f"ce={ce_model or 'none'} (not live; not Reddy 0.857)"
    )
    result = run_ltr_head(
        passages,
        rows,
        dataset_name=path.name,
        k=int(args.k),
        ks=ks,
        n_folds=int(args.folds),
        brain_id="harness-local-ltr",
        source_run=args.from_run,
        pair_policy=policy,
        ce_model=ce_model,
        ce_cache_path=ce_cache,
        ltr_head=head,
        train_eval_result=train_eval,
        train_rows=train_rows,
        train_source_run=train_run or None,
        train_ce_cache_path=train_ce_cache,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    overlap = result.get("overlap_only_metrics") or {}
    console.print(
        f"overlap-only ndcg@10={overlap.get('ndcg@10')} "
        f"recall@10={overlap.get('recall@10')}"
    )
    weights = result.get("ltr_mean_weights") or {}
    if weights:
        kind = "gain importances" if head == "lightgbm" else "weights"
        console.print(f"mean CV {kind} {weights}")
    if result.get("ltr_train_run"):
        console.print(
            f"trained on {result.get('ltr_n_train_queries')} queries "
            f"from {result.get('ltr_train_run')}"
        )
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_pool_first_stage(args: argparse.Namespace, settings: Settings) -> int:
    from search.pool_first_stage import main as pool_main

    argv = ["--dataset", str(_resolve_dataset_path(args, settings))]
    if args.run:
        argv.extend(["--run", str(args.run)])
    if args.variant:
        argv.extend(["--variant", str(args.variant)])
    if args.expand_out:
        argv.extend(["--expand-out", str(args.expand_out)])
    return int(pool_main(argv))


def cmd_rerank_retrieved(args: argparse.Namespace, settings: Settings) -> int:
    from search.rerank_retrieved import run_ce_on_retrieved

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    source_dir = settings.runs_dir / args.from_run
    eval_path = source_dir / "eval.json"
    if not eval_path.exists():
        console.print(f"[red]Missing[/red] {eval_path}")
        return 1
    eval_result = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_result["run_id"] = args.from_run
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    ks = None
    if getattr(args, "ks", None):
        ks = tuple(int(x) for x in str(args.ks).split(",") if x.strip())
    console.print(
        f"[cyan]Harness CE on retrieved hits[/cyan] from={args.from_run} "
        f"dataset={path.name} run={run_id} (not Reddy 0.857)"
    )
    result = run_ce_on_retrieved(
        eval_result,
        rows,
        dataset_name=path.name,
        ks=ks,
        brain_id=settings.brain_id,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    return 0 if report.get("status") == "ok" else 1


def cmd_rank_pool_ce(args: argparse.Namespace, settings: Settings) -> int:
    from search.rank_pool import run_ce_on_pool

    path = _resolve_dataset_path(args, settings)
    if not path.exists():
        console.print(f"[red]Missing[/red] {path}")
        return 1
    rows = load_records(path)
    run_id, run_dir = ensure_run_dir(settings, args.run)
    console.print(
        f"[cyan]Ranking-in-pool CE[/cyan] dataset={path.name} run={run_id} "
        f"(cite Reddy 0.857 only if protocol is CE-on-pool; n is not ~4477)"
    )
    result = run_ce_on_pool(
        rows,
        dataset_name=path.name,
        brain_id=settings.brain_id,
    )
    report = write_report(run_dir, result)
    print_report_table(report)
    console.print(f"[green]Wrote[/green] {run_dir / 'report.json'}")
    if result.get("ce_model"):
        console.print(
            f"ce_model={result.get('ce_model')} "
            f"missing_text={result.get('ce_missing_text')}"
        )
    return 0 if report.get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search",
        description=(
            "Labeled search eval on searchbench*: ingest docs → "
            "POST /retrieve/search (Recall/nDCG/MRR + retrieve latency)"
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
        help=f"Brain id (default {DEFAULT_BRAIN_ID}; must start with searchbench)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser(
        "download",
        help="Download Amazon ESCI and/or WANDS into search JSONL",
    )
    p_download.add_argument(
        "--name",
        default="all",
        help="esci, wands, jdsearch, or all (comma-separated; all = esci,wands)",
    )
    p_download.add_argument("--out", type=str, default=None)
    p_download.add_argument("--force", action="store_true")
    p_download.add_argument(
        "--dry-stats",
        action="store_true",
        dest="dry_stats",
        help="JDsearch only: print label histogram from the tar/cache without writing JSONL",
    )
    p_download.add_argument("--max-queries", type=int, default=80)
    p_download.add_argument("--max-docs", type=int, default=2000)
    p_download.add_argument("--candidates-per-query", type=int, default=40)
    p_download.add_argument(
        "--locale",
        default="us",
        help="ESCI locale: us, es, or jp (default us). Not Italian.",
    )
    p_download.add_argument(
        "--split",
        default="test",
        help="ESCI split (default test; small_version / Task 1 only)",
    )
    p_download.add_argument(
        "--holdout-dataset",
        dest="holdout_dataset",
        default="",
        help="JSONL whose qids are skipped (e.g. data/search_esci_74.jsonl)",
    )
    p_download.set_defaults(func=cmd_download)

    p_stats = sub.add_parser("dataset-stats", help="Summarize search JSONL")
    p_stats.add_argument("--dataset", type=str, default=None)
    p_stats.set_defaults(func=cmd_dataset_stats)

    p_smoke = sub.add_parser(
        "smoke", help="Ingest a few docs then POST /retrieve/search"
    )
    p_smoke.add_argument("--dataset", type=str, default=None)
    p_smoke.add_argument("--limit", type=int, default=2)
    p_smoke.add_argument("--limit-queries", type=int, default=2)
    p_smoke.add_argument("--k", type=int, default=10)
    p_smoke.add_argument("--fusion", choices=["rrf", "cc"], default=None)
    p_smoke.add_argument("--fusion-alpha", dest="fusion_alpha", type=float, default=None)
    p_smoke.add_argument(
        "--rerank",
        default=None,
        help="none or plugin:<name> (e.g. plugin:cross-encoder). Unknown name is 400.",
    )
    p_smoke.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated channels (default passages): "
            "passages,entities,events,communities and/or plugin:<name>."
        ),
    )
    p_smoke.add_argument("--node-labels", dest="node_labels", default=None)
    p_smoke.add_argument("--community-labels", dest="community_labels", default=None)
    p_smoke.add_argument(
        "--expand",
        choices=["none", "neighbors"],
        default="none",
    )
    p_smoke.add_argument(
        "--ingest-graph",
        action="store_true",
        help="Also write deterministic catalog triples (entity uuid = doc_id).",
    )
    p_smoke.add_argument(
        "--interactions",
        type=str,
        default=None,
        help="Optional interaction JSONL (EVENT+happened_at) on this searchbench brain.",
    )
    p_smoke.add_argument("--timeout", type=float, default=600.0)
    p_smoke.add_argument(
        "--enrich",
        action="store_true",
        help="Run Scout/Architect LLM ingest. Default skips enrichment (chunk+embed only).",
    )
    p_smoke.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse chunks already on this searchbench brain (no POST /ingest/).",
    )
    p_smoke.add_argument(
        "--personalize",
        action="store_true",
        dest="personalize",
        help="Send each query's target field on POST /retrieve/search. Default omit.",
    )
    p_smoke.add_argument(
        "--target-from-query",
        action="store_true",
        dest="personalize",
        help="Alias for --personalize.",
    )
    p_smoke.set_defaults(func=cmd_smoke)

    p_bf = sub.add_parser(
        "backfill-entity-text",
        help="ENTITY-only search_text + nodes vector refresh (not architect, not frozen brains)",
    )
    p_bf.add_argument("--dataset", type=str, default="data/search_wands.jsonl")
    p_bf.add_argument("--limit", type=int, default=None)
    p_bf.set_defaults(func=cmd_backfill_entity_text)

    p_eval = sub.add_parser(
        "evaluate",
        help="Ingest corpus, search all queries, write report + ledger",
    )
    p_eval.add_argument("--dataset", type=str, default=None)
    p_eval.add_argument("--run", type=str, default=None)
    p_eval.add_argument("--ks", type=str, default="5,10,20")
    p_eval.add_argument("--k", type=int, default=20)
    p_eval.add_argument("--fusion", choices=["rrf", "cc"], default=None)
    p_eval.add_argument("--fusion-alpha", dest="fusion_alpha", type=float, default=None)
    p_eval.add_argument(
        "--rerank",
        default=None,
        help="none or plugin:<name> (e.g. plugin:cross-encoder). Unknown name is 400.",
    )
    p_eval.add_argument(
        "--mode",
        choices=["default", "catalog"],
        default="default",
        help="default keeps RERANK_MAX_K=10. catalog retrieves deeper and reranks up to 50.",
    )
    p_eval.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated channels (default passages): "
            "passages,entities,events,communities and/or plugin:<name>."
        ),
    )
    p_eval.add_argument("--node-labels", dest="node_labels", default=None)
    p_eval.add_argument("--community-labels", dest="community_labels", default=None)
    p_eval.add_argument(
        "--expand",
        choices=["none", "neighbors"],
        default="none",
    )
    p_eval.add_argument(
        "--ingest-graph",
        action="store_true",
        help="Also write deterministic catalog triples (entity uuid = doc_id).",
    )
    p_eval.add_argument(
        "--interactions",
        type=str,
        default=None,
        help="Optional interaction JSONL (EVENT+happened_at) on this searchbench brain.",
    )
    p_eval.add_argument("--timeout", type=float, default=600.0)
    p_eval.add_argument(
        "--enrich",
        action="store_true",
        help="Run Scout/Architect LLM ingest. Default skips enrichment (chunk+embed only).",
    )
    p_eval.add_argument("--limit-docs", dest="limit_docs", type=int, default=None)
    p_eval.add_argument("--limit-queries", dest="limit_queries", type=int, default=None)
    p_eval.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse chunks already on this searchbench brain (no POST /ingest/).",
    )
    p_eval.add_argument(
        "--rank-pool",
        action="store_true",
        dest="rank_pool",
        help=(
            "Restrict hits to each query's candidate_doc_ids (including I) "
            "and score pool nDCG@20. Do not average with shared-corpus nDCG@10."
        ),
    )
    p_eval.add_argument(
        "--personalize",
        action="store_true",
        dest="personalize",
        help="Send each query's target field on POST /retrieve/search. Default omit.",
    )
    p_eval.add_argument(
        "--target-from-query",
        action="store_true",
        dest="personalize",
        help="Alias for --personalize.",
    )
    p_eval.add_argument(
        "--extras",
        default=None,
        help='Optional equality filter JSON object, e.g. {"locale":"it"}',
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Rebuild report from eval.json")
    p_report.add_argument("--run", type=str, required=True)
    p_report.set_defaults(func=cmd_report)

    p_pool = sub.add_parser(
        "rank-pool-ce",
        help="Score labeled candidate pools with MiniLM CE (no search API)",
    )
    p_pool.add_argument("--dataset", type=str, default=None)
    p_pool.add_argument("--run", type=str, default=None)
    p_pool.set_defaults(func=cmd_rank_pool_ce)

    p_rr = sub.add_parser(
        "rerank-retrieved",
        help="Harness CE over stored first-stage hits (not production RERANK_MAX_K)",
    )
    p_rr.add_argument("--from-run", dest="from_run", type=str, required=True)
    p_rr.add_argument("--dataset", type=str, default=None)
    p_rr.add_argument("--run", type=str, default=None)
    p_rr.add_argument("--ks", type=str, default=None)
    p_rr.set_defaults(func=cmd_rerank_retrieved)

    p_ft = sub.add_parser(
        "finetune-ce",
        help="Fine-tune a pool CE on ESCI US train (test qids held out)",
    )
    p_ft.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ft.add_argument(
        "--out",
        type=str,
        default="data/models/esci-minilm-graded",
    )
    p_ft.add_argument("--max-pairs", type=int, default=80000)
    p_ft.add_argument("--epochs", type=int, default=1)
    p_ft.add_argument("--batch-size", type=int, default=32)
    p_ft.add_argument(
        "--label-mode",
        choices=["binary", "graded", "multiclass"],
        default="graded",
    )
    p_ft.add_argument("--fields", choices=["title", "catalog"], default="catalog")
    p_ft.add_argument(
        "--base",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    p_ft.add_argument("--lr", type=float, default=7e-6)
    p_ft.add_argument("--max-length", type=int, default=256)
    p_ft.set_defaults(func=cmd_finetune_ce)

    p_ms = sub.add_parser(
        "miss-strata",
        help="Classify k=50 hits: head-ok / rank-too-low / total-miss",
    )
    p_ms.add_argument("--from-run", dest="from_run", type=str, required=True)
    p_ms.add_argument("--dataset", type=str, default=None)
    p_ms.add_argument("--out", type=str, default=None)
    p_ms.set_defaults(func=cmd_miss_strata)

    p_qw = sub.add_parser(
        "query-rewrite",
        help="Rewrite pathological total-miss queries only (skip esci-72)",
    )
    p_qw.add_argument("--from-run", dest="from_run", type=str, required=True)
    p_qw.add_argument("--dataset", type=str, default=None)
    p_qw.add_argument("--out", type=str, default=None)
    p_qw.set_defaults(func=cmd_query_rewrite)

    p_sp = sub.add_parser(
        "spell-normalize",
        help="Harness-only NFKC/punctuation/accent normalize of query text",
    )
    p_sp.add_argument("--dataset", type=str, default="data/search_esci_es.jsonl")
    p_sp.add_argument("--out", type=str, default="data/search_esci_es_spell.jsonl")
    p_sp.set_defaults(func=cmd_spell_normalize)

    p_ftd = sub.add_parser(
        "finetune-dense",
        help="Fine-tune a dual encoder with k=50 hard-negative bank (test qids held out)",
    )
    p_ftd.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ftd.add_argument(
        "--out",
        type=str,
        default="data/models/esci-minilm-dense-ance",
    )
    p_ftd.add_argument(
        "--from-eval",
        type=str,
        default="runs/search-esci-74-passages-k50/eval.json",
    )
    p_ftd.add_argument("--max-pairs", type=int, default=20000)
    p_ftd.add_argument("--epochs", type=int, default=1)
    p_ftd.add_argument("--batch-size", type=int, default=32)
    p_ftd.add_argument("--fields", choices=["title", "catalog"], default="catalog")
    p_ftd.add_argument(
        "--base",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p_ftd.add_argument("--lr", type=float, default=2e-5)
    p_ftd.add_argument("--max-length", type=int, default=256)
    p_ftd.set_defaults(func=cmd_finetune_dense)

    p_ld = sub.add_parser(
        "local-dense",
        help="Retrieve the JSONL corpus with a local dual encoder (not BrainAPI)",
    )
    p_ld.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ld.add_argument("--run", type=str, default="search-esci-74-dense-ance-k50")
    p_ld.add_argument("--model", type=str, required=True)
    p_ld.add_argument("--k", type=int, default=50)
    p_ld.add_argument("--ks", type=str, default="5,10,20,50")
    p_ld.set_defaults(func=cmd_local_dense)

    p_mine = sub.add_parser(
        "mine-retrieved-lists",
        help="BM25 top-k train lists with ESCI labels; unlabeled=I",
    )
    p_mine.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_mine.add_argument("--out", type=str, default="data/esci_retrieved_lists.jsonl")
    p_mine.add_argument("--max-queries", type=int, default=6000)
    p_mine.add_argument("--k", type=int, default=50)
    p_mine.add_argument("--seed", type=int, default=11)
    p_mine.set_defaults(func=cmd_mine_retrieved_lists)

    p_exh = sub.add_parser(
        "export-hybrid-lists",
        help="Export stored hybrid k=50 hits as 4-class rows; unlabeled=I",
    )
    p_exh.add_argument(
        "--from-run",
        dest="from_run",
        default="search-esci-ltr200-passages-k50",
    )
    p_exh.add_argument("--dataset", type=str, default="data/search_esci_ltr200.jsonl")
    p_exh.add_argument(
        "--holdout-dataset",
        dest="holdout_dataset",
        default="data/search_esci_74.jsonl",
    )
    p_exh.add_argument("--out", type=str, default="data/esci_hybrid_lists_ltr200.jsonl")
    p_exh.add_argument("--k", type=int, default=50)
    p_exh.set_defaults(func=cmd_export_hybrid_lists)

    p_ft4 = sub.add_parser(
        "finetune-4class",
        help="Fine-tune 4-class L-12 CE (pool or retrieved-bm25 lists)",
    )
    p_ft4.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ft4.add_argument(
        "--out",
        type=str,
        default="data/models/esci-minilm-l12-retrieved",
    )
    p_ft4.add_argument("--from-lists", type=str, default=None)
    p_ft4.add_argument(
        "--lists-source",
        dest="lists_source",
        default="",
        help="Label for --from-lists (hybrid-k50 or retrieved-bm25)",
    )
    p_ft4.add_argument("--max-pairs", type=int, default=80000)
    p_ft4.add_argument("--epochs", type=int, default=1)
    p_ft4.add_argument("--batch-size", type=int, default=32)
    p_ft4.add_argument("--max-length", type=int, default=192)
    p_ft4.add_argument("--ckpt-every", type=int, default=200)
    p_ft4.add_argument("--class-weights", action="store_true")
    p_ft4.add_argument(
        "--base",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
    )
    p_ft4.add_argument("--seed", type=int, default=11)
    p_ft4.set_defaults(func=cmd_finetune_4class)

    p_cb = sub.add_parser(
        "colbert-local",
        help="ColBERT MaxSim sidecar over the JSONL corpus (not BrainAPI)",
    )
    p_cb.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_cb.add_argument("--run", type=str, default="search-esci-74-colbert-k50")
    p_cb.add_argument("--k", type=int, default=50)
    p_cb.add_argument("--ks", type=str, default="5,10,20,50")
    p_cb.set_defaults(func=cmd_colbert_local)

    p_rc = sub.add_parser(
        "rank-corpus",
        help="Score every JSONL doc with 4-class CE (exhaustive-catalog, not production)",
    )
    p_rc.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_rc.add_argument("--run", type=str, default="search-esci-74-exhaustive-ce")
    p_rc.add_argument(
        "--model",
        type=str,
        default="data/models/esci-minilm-l12-4class-nowt-e2",
    )
    p_rc.add_argument("--k", type=int, default=50)
    p_rc.add_argument("--ks", type=str, default="5,10,20,50")
    p_rc.add_argument("--max-length", dest="max_length", type=int, default=192)
    p_rc.set_defaults(func=cmd_rank_corpus)

    p_ov = sub.add_parser(
        "list-overlap",
        help="Count gold ASINs in sidecar top-k missing from passages",
    )
    p_ov.add_argument(
        "--passages-run",
        dest="passages_run",
        default="search-esci-74-passages-k50",
    )
    p_ov.add_argument(
        "--against-runs",
        dest="against_runs",
        default="search-esci-74-bge-base-k50,search-esci-74-colbert-k50",
    )
    p_ov.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ov.add_argument("--run", type=str, default="search-esci-74-list-overlap")
    p_ov.add_argument("--k", type=int, default=50)
    p_ov.set_defaults(func=cmd_list_overlap)

    p_un = sub.add_parser(
        "union-lists",
        help="RRF-union stored first-stage lists (harness only, not live graph)",
    )
    p_un.add_argument(
        "--from-runs",
        dest="from_runs",
        default="search-esci-74-passages-k50,search-esci-74-bge-base-k50",
    )
    p_un.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_un.add_argument("--run", type=str, default="search-esci-74-union-bge-k50")
    p_un.add_argument("--k", type=int, default=50)
    p_un.add_argument("--ks", type=str, default="5,10,20,50")
    p_un.set_defaults(func=cmd_union_lists)

    p_cs = sub.add_parser(
        "cascade-lists",
        help="Keep passages top-10; inject sidecar unique golds into ranks 11-50",
    )
    p_cs.add_argument(
        "--passages-run",
        dest="passages_run",
        default="search-esci-74-passages-k50",
    )
    p_cs.add_argument(
        "--from-runs",
        dest="from_runs",
        default="search-esci-74-bge-base-k50,search-esci-74-colbert-k50",
    )
    p_cs.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_cs.add_argument("--run", type=str, default="search-esci-74-cascade-tail-k50")
    p_cs.add_argument("--k", type=int, default=50)
    p_cs.add_argument("--head-k", dest="head_k", type=int, default=10)
    p_cs.add_argument("--ks", type=str, default="5,10,20,50")
    p_cs.set_defaults(func=cmd_cascade_lists)

    p_ltr = sub.add_parser(
        "ltr-head",
        help="Query-grouped CV LTR on stored hybrid k=50 (BM25/dense rank + title overlap)",
    )
    p_ltr.add_argument(
        "--from-run",
        dest="from_run",
        default="search-esci-74-passages-k50",
    )
    p_ltr.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_ltr.add_argument("--run", type=str, default="search-esci-74-ltr-head-k50")
    p_ltr.add_argument("--k", type=int, default=50)
    p_ltr.add_argument("--ks", type=str, default="5,10,20,50")
    p_ltr.add_argument("--folds", type=int, default=5)
    p_ltr.add_argument(
        "--pair-policy",
        dest="pair_policy",
        default="unlabeled_zero",
        choices=["unlabeled_zero", "other_query_neg"],
    )
    p_ltr.add_argument(
        "--ce-model",
        dest="ce_model",
        default="",
        help="Optional 4-class CE dir; ce_gain is a feature, not the ranker",
    )
    p_ltr.add_argument(
        "--ltr-model",
        dest="ltr_model",
        default="ranknet",
        choices=["ranknet", "lightgbm"],
        help="Linear RankNet (default) or gated LightGBM lambdarank",
    )
    p_ltr.add_argument(
        "--train-from-run",
        dest="train_from_run",
        default="",
        help="Fit on this run's lists; apply to --from-run (no CV on eval qids)",
    )
    p_ltr.add_argument(
        "--train-dataset",
        dest="train_dataset",
        default="",
        help="JSONL for --train-from-run",
    )
    p_ltr.set_defaults(func=cmd_ltr_head)

    p_fs = sub.add_parser(
        "pool-first-stage",
        help="Local fielded BM25 over the JSONL corpus (not Reddy 0.857)",
    )
    p_fs.add_argument("--dataset", type=str, default="data/search_esci_74.jsonl")
    p_fs.add_argument("--run", type=str, default="search-esci-74-fielded-bm25")
    p_fs.add_argument(
        "--variant",
        default="all,title,title-boost,rm3,rm3-title-boost",
    )
    p_fs.add_argument(
        "--expand-out",
        type=str,
        default="data/search_esci_74_rm3.jsonl",
    )
    p_fs.set_defaults(func=cmd_pool_first_stage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    if args.brain:
        settings.brain_id = validate_brain_id(args.brain)
    return int(args.func(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
