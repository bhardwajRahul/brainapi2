import tempfile
import unittest
from pathlib import Path
import os

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


class SearchCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        root = Path(__file__).resolve().parents[1]
        bench = root / "benchmarks"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))

    def test_format_product_text_includes_marker(self):
        from search.catalog import format_product_text
        from search.dataset import doc_marker

        text = format_product_text(
            "B07ABC",
            title="Blue kettle",
            description="1.5 liter stovetop kettle",
            extras=(("Brand", "Acme"), ("Color", "blue")),
        )
        self.assertIn(doc_marker("B07ABC"), text)
        self.assertIn("Title: Blue kettle", text)
        self.assertIn("Brand: Acme", text)

    def test_select_catalog_keeps_distractors_and_grades(self):
        from search.catalog import ESCI_GAINS, _docs_and_queries, _select_catalog

        queries, needed = _select_catalog(
            [
                {
                    "query_id": "10",
                    "query": "blue kettle",
                    "product_id": "p-exact",
                    "esci_label": "E",
                },
                {
                    "query_id": "10",
                    "query": "blue kettle",
                    "product_id": "p-sub",
                    "esci_label": "S",
                },
                {
                    "query_id": "10",
                    "query": "blue kettle",
                    "product_id": "p-irr",
                    "esci_label": "I",
                },
            ],
            max_queries=8,
            max_docs=20,
            candidates_per_query=40,
            slice_name="esci-us",
            dataset="esci",
            gains=ESCI_GAINS,
            label_key="esci_label",
            query_text_key="query",
        )
        self.assertEqual(needed, {"p-exact", "p-sub", "p-irr"})
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["gold_grades"]["p-exact"], 1.0)
        self.assertEqual(queries[0]["gold_grades"]["p-sub"], 0.1)
        self.assertNotIn("p-irr", queries[0]["gold_grades"])
        self.assertEqual(queries[0]["candidate_grades"]["p-irr"], 0.0)
        self.assertEqual(
            queries[0]["candidate_doc_ids"],
            ["p-exact", "p-sub", "p-irr"],
        )
        rows = _docs_and_queries(
            queries,
            products={
                "p-exact": {"product_title": "Blue kettle", "product_description": ""},
                "p-sub": {"product_title": "Teapot", "product_description": ""},
                "p-irr": {"product_title": "Lamp", "product_description": ""},
            },
            doc_ids=needed,
            dataset="esci",
            title_key="product_title",
            description_key="product_description",
        )
        docs = [row for row in rows if row["type"] == "doc"]
        self.assertEqual({row["doc_id"] for row in docs}, needed)
        self.assertEqual(docs[0]["title"], "Blue kettle")

    def test_select_catalog_skips_holdout_qids(self):
        from search.catalog import ESCI_GAINS, _select_catalog

        queries, needed = _select_catalog(
            [
                {
                    "query_id": "72",
                    "query": "held out kettle",
                    "product_id": "p-hold",
                    "esci_label": "E",
                },
                {
                    "query_id": "99",
                    "query": "train sofa",
                    "product_id": "p-train",
                    "esci_label": "E",
                },
            ],
            max_queries=8,
            max_docs=20,
            candidates_per_query=40,
            slice_name="esci-us",
            dataset="esci",
            gains=ESCI_GAINS,
            label_key="esci_label",
            query_text_key="query",
            holdout_qids={"esci-72", "72"},
        )
        self.assertEqual([row["qid"] for row in queries], ["esci-99"])
        self.assertEqual(needed, {"p-train"})

    def test_wands_gains_select_exact_and_partial_not_irrelevant(self):
        from search.catalog import WANDS_GAINS, _select_catalog

        self.assertEqual(
            WANDS_GAINS,
            {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0},
        )
        queries, needed = _select_catalog(
            [
                {
                    "query_id": "1",
                    "query": "velvet sofa",
                    "query_class": "Sofas",
                    "product_id": "p-exact",
                    "label": "Exact",
                },
                {
                    "query_id": "1",
                    "query": "velvet sofa",
                    "query_class": "Sofas",
                    "product_id": "p-partial",
                    "label": "Partial",
                },
                {
                    "query_id": "1",
                    "query": "velvet sofa",
                    "query_class": "Sofas",
                    "product_id": "p-irr",
                    "label": "Irrelevant",
                },
            ],
            max_queries=8,
            max_docs=20,
            candidates_per_query=40,
            slice_name="wands",
            dataset="wands",
            gains=WANDS_GAINS,
            label_key="label",
            query_text_key="query",
            query_slice_key="query_class",
        )
        self.assertEqual(needed, {"p-exact", "p-partial", "p-irr"})
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["qid"], "wands-1")
        self.assertEqual(queries[0]["gold_grades"]["p-exact"], 1.0)
        self.assertEqual(queries[0]["gold_grades"]["p-partial"], 0.5)
        self.assertNotIn("p-irr", queries[0]["gold_grades"])
        self.assertEqual(queries[0]["gold_doc_ids"], ["p-exact", "p-partial"])
        self.assertEqual(queries[0]["candidate_grades"]["p-irr"], 0.0)

    def test_frozen_wands_jsonl_blocks_overwrite(self):
        from search.catalog import DATA_DIR, catalog_overwrite_blocked

        self.assertTrue(
            catalog_overwrite_blocked(DATA_DIR / "search_esci_74.jsonl")
        )
        wands = DATA_DIR / "search_wands.jsonl"
        self.assertTrue(wands.exists() and wands.stat().st_size > 0)
        self.assertTrue(catalog_overwrite_blocked(wands))
        self.assertFalse(
            catalog_overwrite_blocked(DATA_DIR / "search_esci.jsonl")
        )

    def test_frozen_structured_brains_exclude_wandsgraph(self):
        from search.config import FROZEN_STRUCTURED_BRAINS

        self.assertIn("searchbenchwands", FROZEN_STRUCTURED_BRAINS)
        self.assertIn("searchbenchesci74", FROZEN_STRUCTURED_BRAINS)
        self.assertNotIn("searchbenchwandsgraph", FROZEN_STRUCTURED_BRAINS)
        self.assertNotIn("searchbenchjdslice", FROZEN_STRUCTURED_BRAINS)

    def test_evaluate_refuses_frozen_structured_ingest(self):
        from search.evaluate import evaluate_search

        class _Client:
            class _Settings:
                brain_id = "searchbenchwands"

            settings = _Settings()

        with self.assertRaises(SystemExit):
            evaluate_search(_Client(), [], ingest_graph=True)
        with self.assertRaises(SystemExit):
            evaluate_search(
                _Client(),
                [],
                interactions=[{"user_id": "u", "item_id": "0"}],
            )

    def test_wands_feature_parse_and_catalog_triples(self):
        from search.mapping import (
            docs_to_triples,
            entity_uuid,
            parse_feature_string,
        )

        parsed = parse_feature_string(
            "|Color:Navy|Material:Velvet|InStock:yes|Finish:|"
        )
        self.assertEqual(parsed, [("Color", "Navy"), ("Material", "Velvet")])
        docs = [
            {
                "doc_id": f"d{i}",
                "title": f"Item {i}",
                "class": "sofas" if i == 1 else "",
                "brand": "Acme" if i == 2 else "",
                "features": "|Color:Navy|" if i == 3 else "",
                "dataset": "wands",
            }
            for i in range(1, 6)
        ]
        triples = docs_to_triples(docs)
        uuids = {row["subject"]["uuid"] for row in triples}
        self.assertEqual(uuids, {entity_uuid(f"d{i}") for i in range(1, 6)})
        self.assertTrue(all(row.get("event") is None for row in triples))
        self.assertTrue(all(row["subj_event"]["name"] == "HAS" for row in triples))
        self.assertTrue(all("event_obj" not in row or row.get("event_obj") is None for row in triples))
        objects = {(row["object"]["type"], row["object"]["name"]) for row in triples}
        self.assertIn(("CLASS", "sofas"), objects)
        self.assertIn(("ATTR", "Acme"), objects)
        self.assertIn(("ATTR", "Navy"), objects)
        self.assertNotIn("demorecsys", str(triples))

    def test_catalog_graph_sofas_modern_has_search_text(self):
        from search.mapping import doc_to_triples, parse_feature_string, split_hierarchy

        parsed = parse_feature_string("dsprimaryproductstyle : modern|color : navy")
        self.assertIn(("dsprimaryproductstyle", "modern"), parsed)
        self.assertEqual(split_hierarchy("Furniture > Living Room > Sofas"), ["Furniture", "Living Room", "Sofas"])
        triples = doc_to_triples(
            {
                "doc_id": "sofa-1",
                "title": "Velvet sofa",
                "description": "A low modern sofa for small rooms",
                "class": "sofas",
                "hierarchy": "Furniture > Living Room > Sofas",
                "features": "style:modern|color:navy",
                "price": "899",
                "dataset": "wands",
            }
        )
        objects = {(row["object"]["type"], row["object"]["name"]) for row in triples}
        self.assertIn(("CLASS", "sofas"), objects)
        self.assertIn(("ATTR", "modern"), objects)
        self.assertIn(("TYPE", "Furniture"), objects)
        subject = triples[0]["subject"]
        self.assertIn("modern sofa", subject["properties"]["search_text"].lower())
        self.assertIn("sofas", subject["properties"]["search_text"].lower())
        self.assertIn("modern", subject["properties"]["search_text"].lower())
        self.assertIn("navy", subject["properties"]["search_text"].lower())
        self.assertEqual(subject["properties"]["price"], "899")
        hub_texts = {row["object"]["properties"]["search_text"] for row in triples}
        self.assertTrue(any("modern" in text.lower() for text in hub_texts))
        self.assertNotIn(("ATTR", "899"), objects)
        self.assertTrue(all(row.get("event") is None for row in triples))
        self.assertTrue(all(row["subj_event"]["name"] == "HAS" for row in triples))
        self.assertTrue(
            all("dsprimaryproductstyle" not in str(row.get("subj_event") or {}) for row in triples)
        )

    def test_entity_search_text_prefers_catalog_blob(self):
        from search.mapping import doc_id_from_text, doc_to_triples, node_id_from_passage_text

        blob = (
            "DOCID wands-bed.\n"
            "Title: solid wood platform bed\n"
            "Class: Beds\n"
            "Features: dsprimaryproductstyle : modern|woodspecies : rubberwood"
        )
        triples = doc_to_triples(
            {
                "doc_id": "wands-bed",
                "title": "solid wood platform bed",
                "class": "Beds",
                "features": "dsprimaryproductstyle : modern|woodspecies : rubberwood",
                "text": blob,
            }
        )
        subject = triples[0]["subject"]
        self.assertEqual(subject["properties"]["search_text"], blob)
        self.assertIn("rubberwood", subject["properties"]["search_text"])
        self.assertIsNone(triples[0].get("event"))
        self.assertEqual(triples[0]["subj_event"]["name"], "HAS")
        self.assertNotIn("rubberwood", triples[0]["subj_event"]["name"])
        self.assertEqual(doc_id_from_text(blob), "wands-bed")
        self.assertEqual(node_id_from_passage_text(blob), "wands-bed")
        self.assertIsNone(node_id_from_passage_text("no marker here"))

    def test_catalog_has_is_direct_edge(self):
        from search.mapping import doc_to_triples, is_static_has_triple

        triples = doc_to_triples(
            {
                "doc_id": "0",
                "title": "solid wood platform bed",
                "class": "Beds",
                "features": "woodspecies : rubberwood",
            }
        )
        self.assertTrue(triples)
        self.assertTrue(all(is_static_has_triple(row) for row in triples))
        self.assertTrue(all(row.get("event") is None for row in triples))
        kinds = {(row["object"]["type"], row["object"]["name"]) for row in triples}
        self.assertIn(("CLASS", "Beds"), kinds)
        self.assertIn(("ATTR", "rubberwood"), kinds)

    def test_entity_backfill_refuses_frozen_brains(self):
        from search.backfill_entities import (
            apply_entity_text_backfill,
            backfill_rows_from_docs,
            refuse_entity_backfill,
        )

        self.assertEqual(refuse_entity_backfill("searchbenchwandsgraph"), "searchbenchwandsgraph")
        with self.assertRaises(SystemExit):
            refuse_entity_backfill("searchbenchwands")
        with self.assertRaises(SystemExit):
            refuse_entity_backfill("searchbenchesci74")
        rows = backfill_rows_from_docs(
            [
                {
                    "doc_id": "0",
                    "title": "solid wood platform bed",
                    "class": "Beds",
                    "features": "dsprimaryproductstyle : modern|woodspecies : rubberwood",
                    "text": "DOCID 0.\nTitle: solid wood platform bed\nFeatures: woodspecies : rubberwood",
                }
            ]
        )
        self.assertEqual(rows[0]["uuid"], "0")
        self.assertIn("rubberwood", rows[0]["search_text"])
        with self.assertRaises(SystemExit):
            apply_entity_text_backfill(
                brain_id="searchbenchwands",
                rows=rows,
                graph=None,
                embeddings=None,
                vector_store=None,
            )

    def test_apply_entity_backfill_updates_entity_not_event(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from search.backfill_entities import apply_entity_text_backfill
        from src.constants.embeddings import Vector

        node = SimpleNamespace(
            uuid="0",
            name="bed",
            labels=["ENTITY"],
            properties={"search_text": "bed", "v_id": "old-v"},
        )
        graph = MagicMock()
        graph.get_by_uuid.return_value = node
        embeddings = MagicMock()
        embeddings.embed_text.return_value = Vector(
            id="new", embeddings=[0.1], metadata={}
        )
        store = MagicMock()
        store.add_vectors.return_value = ["vid-1"]
        summary = apply_entity_text_backfill(
            brain_id="searchbenchwandsgraph",
            rows=[{"uuid": "0", "name": "bed", "search_text": "rubberwood modern bed"}],
            graph=graph,
            embeddings=embeddings,
            vector_store=store,
            is_item=lambda uuid, labels: "ENTITY" in (labels or []),
        )
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["missing"], 0)
        written = [
            call.kwargs.get("new_properties") or {}
            for call in graph.update_node.call_args_list
        ]
        self.assertTrue(any("rubberwood" in str(props.get("search_text") or "") for props in written))
        event = SimpleNamespace(
            uuid="evt-1",
            name="HAS",
            labels=["EVENT"],
            properties={},
        )
        graph.get_by_uuid.return_value = event
        skipped = apply_entity_text_backfill(
            brain_id="searchbenchwandsgraph",
            rows=[{"uuid": "evt-1", "name": "HAS", "search_text": "should not copy"}],
            graph=graph,
            embeddings=embeddings,
            vector_store=store,
            is_item=lambda uuid, labels: "ENTITY" in (labels or []),
        )
        self.assertEqual(skipped["updated"], 0)
        self.assertEqual(skipped["skipped"], 1)

    def test_wandsgraph_node_join_calls_neighbors(self):
        from search.client import TimedResult
        from search.evaluate import assert_wandsgraph_node_join

        class _Client:
            class _Settings:
                brain_id = "searchbenchwandsgraph"

            settings = _Settings()

            def get_neighbors(self, uuid, limit=5):
                self.seen = uuid
                return TimedResult(data={"count": 4}, latency_ms=1.0, status_code=200)

        client = _Client()
        out = assert_wandsgraph_node_join(
            client,
            {
                "queries": [
                    {
                        "qid": "q1",
                        "hits": [
                            {
                                "id": "chunk-1",
                                "doc_id": "0",
                                "node_id": "0",
                                "channel": "passages",
                            }
                        ],
                    }
                ]
            },
        )
        self.assertEqual(out["node_id"], "0")
        self.assertEqual(out["status_code"], 200)
        self.assertEqual(client.seen, "0")
        skipped = assert_wandsgraph_node_join(
            type("C", (), {"settings": type("S", (), {"brain_id": "searchbenchsmoke"})()})(),
            {"queries": []},
        )
        self.assertTrue(skipped["skipped"])

    def test_interaction_triples_have_happened_at(self):
        from search.mapping import interactions_to_triples, load_interaction_rows

        rows = [
            {
                "user_id": "u01",
                "item_id": "sku-101",
                "behavior": "view",
                "timestamp": "2024-01-02T10:00:00Z",
                "category": "phones",
                "brand": "Acme",
            },
            {
                "user_id": "u01",
                "item_id": "sku-102",
                "behavior": "purchase",
                "timestamp": "2024-01-05T12:00:00Z",
            },
            {
                "user_id": "u02",
                "item_id": "sku-201",
                "behavior": "view",
                "timestamp": "2024-01-02T09:00:00Z",
            },
        ]
        triples = interactions_to_triples(rows)
        events = [
            row["event"]
            for row in triples
            if row.get("event") and row["event"]["name"] != "HAS"
        ]
        self.assertGreaterEqual(len(events), 3)
        self.assertTrue(all(event.get("happened_at") for event in events))
        self.assertTrue(all(event["type"] == "EVENT" for event in events))
        self.assertTrue(all(row["object"]["uuid"] in {"sku-101", "sku-102", "sku-201"} or row["object"]["type"] != "ENTITY" for row in triples))
        entity_ids = {
            row["object"]["uuid"]
            for row in triples
            if row["object"]["type"] == "ENTITY"
        }
        self.assertEqual(entity_ids, {"sku-101", "sku-102", "sku-201"})
        self.assertNotIn("PRODUCT", str(triples))
        self.assertNotIn("Categorized", str(triples))
        self.assertNotIn("demorecsys", str(triples))
        root = Path(__file__).resolve().parents[1]
        toy = root / "benchmarks" / "data" / "recsys_toy.jsonl"
        loaded = load_interaction_rows(toy)[:3]
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["item_id"], "sku-101")

    def test_interaction_options_emit_prefers_not_catalog_has(self):
        from search.mapping import hub_uuid, interaction_to_triples

        triples = interaction_to_triples(
            {
                "user_id": "u01",
                "item_id": "sku-101",
                "behavior": "view",
                "timestamp": "2024-01-02T10:00:00Z",
                "color": "navy",
                "options": {"style": "70s", "color": "olive"},
            },
            seq=1,
        )
        prefers = [
            row
            for row in triples
            if (row.get("subj_event") or {}).get("name") == "PREFERS"
        ]
        self.assertEqual(len(prefers), 2)
        self.assertTrue(all(row.get("event") is None for row in prefers))
        self.assertTrue(all(row["subject"]["uuid"] == "user:u01" for row in prefers))
        style_hub = hub_uuid("attr", "70s")
        olive_hub = hub_uuid("attr", "olive")
        prefers_tips = {row["object"]["uuid"] for row in prefers}
        self.assertEqual(prefers_tips, {style_hub, olive_hub})
        has_tips = {
            row["object"]["uuid"]
            for row in triples
            if (row.get("subj_event") or {}).get("name") == "HAS"
        }
        self.assertIn(hub_uuid("attr", "navy"), has_tips)
        self.assertNotIn(style_hub, has_tips)
        self.assertNotIn(olive_hub, has_tips)

    def test_personalize_smoke_fixture_emits_prefers(self):
        from search.mapping import hub_uuid, load_interaction_rows, interactions_to_triples

        root = Path(__file__).resolve().parents[1]
        rows = load_interaction_rows(
            root / "benchmarks" / "data" / "search_personalize_smoke.jsonl"
        )
        self.assertEqual(len(rows), 4)
        triples = interactions_to_triples(rows)
        prefers = [
            row
            for row in triples
            if (row.get("subj_event") or {}).get("name") == "PREFERS"
        ]
        self.assertGreaterEqual(len(prefers), 1)
        self.assertIn(hub_uuid("attr", "70s"), {row["object"]["uuid"] for row in prefers})
        self.assertTrue(any(row["event"]["name"] == "Favorite" for row in triples if row.get("event")))

    def test_cli_unknown_catalog(self):
        from search.cli import main

        self.assertEqual(main(["download", "--name", "not-a-dataset"]), 1)

    def test_write_roundtrip(self):
        from search.dataset import dataset_stats, load_records, write_records

        rows = [
            {"type": "doc", "doc_id": "a", "text": "DOCID a. Title: kettle"},
            {
                "type": "query",
                "qid": "q1",
                "query": "kettle",
                "gold_doc_ids": ["a"],
                "gold_grades": {"a": 1.0},
                "slice": "esci-us",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_esci.jsonl"
            write_records(rows, path)
            loaded = load_records(path)
            stats = dataset_stats(loaded)
        self.assertEqual(stats["n_docs"], 1)
        self.assertEqual(stats["n_queries"], 1)
        self.assertTrue(stats["graded"])

    def test_esci_locale_paths_keep_us_default(self):
        from search.catalog import ESCI_JSONL, catalog_jsonl_path, normalize_esci_locale

        self.assertEqual(catalog_jsonl_path("esci"), ESCI_JSONL)
        self.assertEqual(catalog_jsonl_path("esci", locale="us"), ESCI_JSONL)
        self.assertEqual(catalog_jsonl_path("esci", locale="US"), ESCI_JSONL)
        es_path = catalog_jsonl_path("esci", locale="es")
        jp_path = catalog_jsonl_path("esci", locale="jp")
        self.assertEqual(es_path.name, "search_esci_es.jsonl")
        self.assertEqual(jp_path.name, "search_esci_jp.jsonl")
        self.assertNotEqual(es_path, ESCI_JSONL)
        self.assertNotEqual(jp_path, ESCI_JSONL)
        self.assertEqual(normalize_esci_locale("es"), "es")

    def test_esci_locale_rejects_italian(self):
        from search.catalog import catalog_jsonl_path, normalize_esci_locale
        from search.cli import main

        with self.assertRaises(ValueError) as ctx:
            normalize_esci_locale("it")
        self.assertIn("no Italian", str(ctx.exception))
        with self.assertRaises(ValueError):
            catalog_jsonl_path("esci", locale="it")
        self.assertEqual(main(["download", "--name", "esci", "--locale", "it"]), 1)

    def test_italian_smoke_is_not_esci(self):
        from search.dataset import dataset_stats, load_records

        root = Path(__file__).resolve().parents[1]
        path = root / "benchmarks" / "data" / "search_italian_smoke.jsonl"
        rows = load_records(path)
        stats = dataset_stats(rows)
        self.assertGreaterEqual(stats["n_docs"], 3)
        self.assertGreaterEqual(stats["n_queries"], 3)
        self.assertEqual(stats["slices"], {"italian-smoke": stats["n_queries"]})
        for row in rows:
            self.assertNotEqual(row.get("dataset"), "esci")
            self.assertNotIn("esci-", str(row.get("qid") or ""))
            self.assertNotIn("esci-", str(row.get("slice") or ""))

    def test_italian_inflect_is_not_esci(self):
        from search.dataset import dataset_stats, load_records

        root = Path(__file__).resolve().parents[1]
        path = root / "benchmarks" / "data" / "search_italian_smoke_inflect.jsonl"
        rows = load_records(path)
        stats = dataset_stats(rows)
        self.assertGreaterEqual(stats["n_docs"], 3)
        self.assertGreaterEqual(stats["n_queries"], 3)
        self.assertEqual(stats["slices"], {"italian-smoke-inflect": stats["n_queries"]})
        queries = [row["query"] for row in rows if row.get("type") == "query"]
        self.assertIn("bollitori acciaio", queries)
        self.assertIn("divani velluto", queries)
        self.assertIn("caffettiere alluminio", queries)
        for row in rows:
            self.assertNotEqual(row.get("dataset"), "esci")
            self.assertNotIn("esci-", str(row.get("qid") or ""))
            self.assertNotIn("esci-", str(row.get("slice") or ""))


if __name__ == "__main__":
    unittest.main()
