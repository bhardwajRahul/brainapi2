import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jdsearch"
)


class SearchJdsearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        root = Path(__file__).resolve().parents[1]
        bench = root / "benchmarks"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))

    def _fixture_paths(self) -> tuple[Path, Path]:
        product = FIXTURE_DIR / "product_meta_data.txt"
        behavior = FIXTURE_DIR / "user_behavior_data.txt"
        self.assertTrue(product.exists(), product)
        self.assertTrue(behavior.exists(), behavior)
        return product, behavior

    def test_term_and_list_separators(self):
        from search.jdsearch import TERM_SEP, join_terms, split_list

        self.assertEqual(TERM_SEP, "\030")
        self.assertEqual(join_terms(f"111{TERM_SEP}222"), "111 222")
        self.assertEqual(split_list("100_200_999"), ["100", "200", "999"])
        self.assertEqual(split_list(""), [])

    def test_stats_histogram_and_truncated_count(self):
        from search.jdsearch import PAPER_N_PRODUCTS, collect_jdsearch_stats

        product, behavior = self._fixture_paths()
        stats = collect_jdsearch_stats(
            product_path=product,
            behavior_path=behavior,
            extract=False,
        )
        self.assertEqual(stats["n_behavior_rows"], 2)
        self.assertEqual(stats["n_products_seen"], 2)
        self.assertFalse(stats["truncated"])
        self.assertIn("1", stats["label_histogram"])
        self.assertEqual(stats["label_scheme"], "graded")
        self.assertGreater(stats["history_type_counts"].get("CLICK", 0), 0)
        self.assertGreater(stats["history_type_counts"].get("FLW", 0), 0)
        self.assertIn("999", {"999"})
        self.assertGreater(stats["n_wids_missing_meta"], 0)

        truncated = collect_jdsearch_stats(
            product_path=product,
            behavior_path=behavior,
            extract=False,
            max_behavior_rows=1,
            max_product_rows=1,
        )
        self.assertTrue(truncated["truncated"])
        self.assertEqual(truncated["n_behavior_rows"], 1)
        self.assertEqual(truncated["n_products_seen"], 1)
        self.assertNotEqual(truncated["n_products_seen"], PAPER_N_PRODUCTS)
        self.assertLess(truncated["n_products_seen"], 12_000_000)

    def test_prepare_rows_caps_target_and_gold(self):
        from search.catalog import FROZEN_JSONL_IF_EXISTS, prepare_catalog
        from search.dataset import dataset_stats, load_records, split_corpus
        from search.jdsearch import (
            JDSEARCH_NAME,
            jdsearch_interactions_path,
            prepare_jdsearch_rows,
        )

        product, behavior = self._fixture_paths()
        wands = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "search_wands.jsonl"
        )
        wands_mtime = wands.stat().st_mtime if wands.exists() else None
        rows, interactions, stats = prepare_jdsearch_rows(
            max_queries=80,
            max_docs=2000,
            product_path=product,
            behavior_path=behavior,
            extract=False,
        )
        docs, queries = split_corpus(rows)
        self.assertLessEqual(len(queries), 80)
        self.assertLessEqual(len(docs), 2000)
        self.assertEqual(stats["label_scheme"], "graded")
        self.assertTrue(queries)
        for query in queries:
            self.assertTrue(str(query.get("target") or "").startswith("jd-u"))
            self.assertTrue(query.get("gold_doc_ids"))
            self.assertTrue(query.get("gold_grades"))
            self.assertTrue(query.get("candidate_doc_ids"))
        self.assertTrue(any(row.get("behavior") == "follow" for row in interactions))
        self.assertTrue(any(row.get("behavior") == "click" for row in interactions))
        qids = {query["qid"] for query in queries}
        self.assertTrue(qids)
        for query in queries:
            self.assertNotIn("999", query["gold_doc_ids"])
        self.assertIn("search_jdsearch.jsonl", FROZEN_JSONL_IF_EXISTS)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "search_jdsearch.jsonl"
            written = prepare_catalog(
                JDSEARCH_NAME,
                out_path=dest,
                max_queries=80,
                max_docs=2000,
            )
            self.assertEqual(written, dest)
            loaded = load_records(dest)
            stats = dataset_stats(loaded)
            self.assertLessEqual(stats["n_queries"], 80)
            self.assertLessEqual(stats["n_docs"], 2000)
            _, loaded_queries = split_corpus(loaded)
            self.assertTrue(all(q.get("target") and q.get("gold_doc_ids") for q in loaded_queries))
            inter_path = jdsearch_interactions_path(dest)
            self.assertTrue(inter_path.exists())
            self.assertFalse((Path(tmp) / "search_wands.jsonl").exists())
            self.assertFalse((Path(tmp) / "search_esci_74.jsonl").exists())
        if wands_mtime is not None:
            self.assertEqual(wands.stat().st_mtime, wands_mtime)

    def test_prepare_drops_all_missing_gold(self):
        from search.jdsearch import prepare_jdsearch_rows

        product, behavior = self._fixture_paths()
        with tempfile.TemporaryDirectory() as tmp:
            extra_behavior = Path(tmp) / "user_behavior_data.txt"
            extra_behavior.write_text(
                behavior.read_text(encoding="utf-8")
                + "onlymissing\t999\t1.0\t-1\t\t\t0\n",
                encoding="utf-8",
            )
            rows, _interactions, _stats = prepare_jdsearch_rows(
                product_path=product,
                behavior_path=extra_behavior,
                extract=False,
            )
            queries = [row for row in rows if row.get("type") == "query"]
            self.assertTrue(all("999" not in (q.get("gold_doc_ids") or []) for q in queries))
            self.assertTrue(all(q.get("gold_doc_ids") for q in queries))

    def test_follow_event_not_has(self):
        import importlib.util

        from search.mapping import HAS_EVENT_NAME, interaction_to_triples

        triples = interaction_to_triples(
            [
                {
                    "user_id": "jd-u0",
                    "item_id": "100",
                    "behavior": "follow",
                    "timestamp": "2022-10-16T00:00:00Z",
                }
            ]
        )
        events = [
            row["event"]["name"]
            for row in triples
            if isinstance(row.get("event"), dict) and row["event"].get("name")
        ]
        self.assertIn("Follow", events)
        self.assertNotIn(HAS_EVENT_NAME, events)
        mapping_path = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "features-rec"
            / "models"
            / "mapping.py"
        )
        spec = importlib.util.spec_from_file_location(
            "features_rec_mapping_jdsearch", mapping_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.normalize_behavior("FLW"), ("Follow", "TARGETED"))
        self.assertEqual(mod.behavior_weight("FLW"), 0.5)
        self.assertEqual(mod.behavior_weight("follow"), 0.5)

    def test_evaluate_sends_target_only_when_personalized(self):
        from search.client import TimedResult
        from search.evaluate import evaluate_search
        from src.services.api.constants.requests import SearchRequestBody

        fields = set(SearchRequestBody.model_fields)
        self.assertIn("target", fields)
        self.assertNotIn("user_id", fields)

        class _Client:
            class _Settings:
                brain_id = "searchbenchjdslice"

            settings = _Settings()

            def __init__(self):
                self.calls = []

            def search(self, query, **kwargs):
                self.calls.append({"query": query, **kwargs})
                return TimedResult(data={"hits": []}, latency_ms=1.0)

            def list_text_chunks(self, **kwargs):
                skip = int(kwargs.get("skip") or 0)
                if skip:
                    return TimedResult(data={"data": [], "total": 1}, latency_ms=1.0)
                return TimedResult(
                    data={
                        "data": [{"id": "c1", "text": "DOCID 100. 111 222"}],
                        "total": 1,
                    },
                    latency_ms=1.0,
                )

        rows = [
            {
                "type": "doc",
                "doc_id": "100",
                "text": "DOCID 100. 111 222",
            },
            {
                "type": "query",
                "qid": "jdsearch-0",
                "query": "111 222",
                "target": "jd-u0",
                "gold_doc_ids": ["100"],
                "gold_grades": {"100": 1.0},
                "candidate_doc_ids": ["100"],
            },
        ]
        client = _Client()
        evaluate_search(client, rows, skip_ingest=True, personalize=True)
        self.assertEqual(client.calls[0].get("target"), "jd-u0")
        client.calls.clear()
        evaluate_search(client, rows, skip_ingest=True, personalize=False)
        self.assertFalse(client.calls[0].get("target"))
        client.calls.clear()
        evaluate_search(
            client,
            [{**rows[1], "target": ""}, rows[0]]
            if False
            else [
                rows[0],
                {**rows[1], "target": ""},
            ],
            skip_ingest=True,
            personalize=True,
        )
        self.assertFalse(client.calls[0].get("target"))

    def test_client_search_body_omits_target_by_default(self):
        from search.client import BrainAPIClient, TimedResult
        from search.config import Settings

        captured = {}

        class _Stub(BrainAPIClient):
            def __post_init__(self):
                self._client = MagicMock()

            def _request(self, method, path, *, json=None, params=None):
                captured["json"] = json
                return TimedResult(data={"hits": []}, latency_ms=1.0)

        settings = MagicMock(spec=Settings)
        settings.brain_id = "searchbenchjdslice"
        settings.brainapi_url = "http://127.0.0.1"
        settings.brainpat_token = "token"
        settings.require_brainapi = lambda: None
        client = _Stub(settings)
        client.search("111 222")
        self.assertNotIn("target", captured["json"])
        client.search("111 222", target="jd-u0")
        self.assertEqual(captured["json"]["target"], "jd-u0")
        self.assertNotIn("user_id", captured["json"])

    def test_bm25_tokens_present_in_doc_text(self):
        from search.jdsearch import prepare_jdsearch_rows

        product, behavior = self._fixture_paths()
        rows, _interactions, _stats = prepare_jdsearch_rows(
            product_path=product,
            behavior_path=behavior,
            extract=False,
        )
        docs = [row for row in rows if row.get("type") == "doc"]
        queries = [row for row in rows if row.get("type") == "query"]
        self.assertTrue(docs)
        blob = " ".join(str(doc.get("text") or "") for doc in docs)
        self.assertIn("DOCID 100.", blob)
        nonempty = 0
        for query in queries:
            tokens = str(query.get("query") or "").split()
            if any(token and token in blob for token in tokens):
                nonempty += 1
        self.assertGreater(nonempty, 0)
