import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SearchBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        root = Path(__file__).resolve().parents[1]
        bench = root / "benchmarks"
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))

    def require_local_artifact(self, path: Path) -> Path:
        if not path.is_file():
            self.skipTest(f"optional local benchmark artifact is absent: {path}")
        return path

    def test_brain_id_guard(self):
        from search.config import DEFAULT_BRAIN_ID, validate_brain_id

        self.assertEqual(validate_brain_id("searchbenchsmoke"), "searchbenchsmoke")
        self.assertEqual(validate_brain_id(DEFAULT_BRAIN_ID), DEFAULT_BRAIN_ID)
        self.assertEqual(validate_brain_id("searchbench1"), "searchbench1")
        self.assertEqual(validate_brain_id("searchbenchjdslice"), "searchbenchjdslice")
        for bad in (
            "locomoconv26",
            "locomoconv26clean",
            "beam1m1clean",
            "beam100k1",
            "demorecsys",
            "lme-s-smoke",
            "default",
            "search",
        ):
            with self.assertRaises(SystemExit):
                validate_brain_id(bad)

    def test_metrics_ranking_and_latency(self):
        from search.metrics import (
            aggregate_query_metrics,
            mrr,
            ndcg_at_k,
            percentile,
            recall_at_k,
            retrieve_latency_ms,
        )

        ranked = ["a", "b", "gold", "d"]
        gold = {"gold"}
        self.assertEqual(recall_at_k(ranked, gold, 2), 0.0)
        self.assertEqual(recall_at_k(ranked, gold, 3), 1.0)
        self.assertEqual(mrr(ranked, gold), 1.0 / 3)
        self.assertGreater(ndcg_at_k(ranked, gold, 10), 0.0)
        self.assertLess(ndcg_at_k(ranked, gold, 10), 1.0)
        self.assertEqual(ndcg_at_k(["gold"], gold, 10), 1.0)
        self.assertGreater(
            ndcg_at_k(
                ["exact", "sub"],
                {"exact", "sub"},
                10,
                grades={"exact": 1.0, "sub": 0.1},
            ),
            ndcg_at_k(
                ["sub", "exact"],
                {"exact", "sub"},
                10,
                grades={"exact": 1.0, "sub": 0.1},
            ),
        )
        wands_gold = {"exact", "partial"}
        wands_grades = {"exact": 1.0, "partial": 0.5}
        self.assertGreater(
            ndcg_at_k(["exact", "partial"], wands_gold, 10, grades=wands_grades),
            ndcg_at_k(["partial", "exact"], wands_gold, 10, grades=wands_grades),
        )
        self.assertAlmostEqual(
            ndcg_at_k(["exact", "partial"], wands_gold, 10, grades=wands_grades),
            ndcg_at_k(
                ["exact", "partial"],
                wands_gold,
                10,
                grades={"exact": 2.0, "partial": 1.0},
            ),
        )
        import math

        from search.metrics import _dcg

        self.assertAlmostEqual(
            _dcg([1.0, 0.5], 2),
            1.0 / math.log2(2) + 0.5 / math.log2(3),
        )
        self.assertEqual(recall_at_k(["exact", "other"], wands_gold, 10), 0.5)
        self.assertEqual(percentile([10.0, 20.0, 30.0], 50), 20.0)

        retrieve, embed = retrieve_latency_ms(
            {
                "stages": [
                    {"stage": "embed.query", "wall_ms": 80.0},
                    {"stage": "search.retrieve", "wall_ms": 12.5},
                ]
            },
            200.0,
        )
        self.assertEqual(retrieve, 12.5)
        self.assertEqual(embed, 80.0)
        retrieve_fallback, embed_fallback = retrieve_latency_ms(
            {"stages": [{"stage": "embed.query", "wall_ms": 40.0}]},
            90.0,
        )
        self.assertEqual(retrieve_fallback, 50.0)
        self.assertEqual(embed_fallback, 40.0)

        rows = [
            {
                "slice": "keyword",
                "metrics": {
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "recall@20": 1.0,
                    "ndcg@10": 1.0,
                    "ndcg@20": 1.0,
                    "ndcg": 1.0,
                    "mrr": 1.0,
                },
                "retrieve_ms": 10.0,
                "embed_ms": 50.0,
                "client_wall_ms": 70.0,
            },
            {
                "slice": "paraphrase",
                "metrics": {
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "recall@20": 1.0,
                    "ndcg@10": 0.0,
                    "ndcg@20": 0.5,
                    "ndcg": 0.5,
                    "mrr": 0.0,
                },
                "retrieve_ms": 30.0,
                "embed_ms": 50.0,
                "client_wall_ms": 90.0,
            },
        ]
        metrics = aggregate_query_metrics(rows)
        self.assertEqual(metrics["recall@10"], 0.5)
        self.assertEqual(metrics["ndcg@10"], 0.5)
        self.assertEqual(metrics["ndcg@20"], 0.75)
        self.assertEqual(metrics["by_slice"]["keyword"]["recall@10"], 1.0)
        self.assertEqual(metrics["p50_retrieve_ms"], 20.0)

        from search.metrics import ndcg_at_k

        ranked_pool = ["exact", "irr", "sub", "other"]
        grades_pool = {"exact": 1.0, "sub": 0.1, "irr": 0.0}
        gold_pool = {"exact", "sub"}
        manual = ndcg_at_k(ranked_pool, gold_pool, 20, grades=grades_pool)
        self.assertGreater(manual, 0.0)
        self.assertEqual(
            ndcg_at_k(ranked_pool, gold_pool, 20, grades=grades_pool),
            ndcg_at_k(ranked_pool, gold_pool, 4, grades=grades_pool),
        )

    def test_doc_marker_mapping(self):
        from search.dataset import dataset_stats, load_records, map_doc_ids_to_chunks

        root = Path(__file__).resolve().parents[1]
        fixture = self.require_local_artifact(
            root / "benchmarks" / "data" / "search_toy.jsonl"
        )
        rows = load_records(fixture)
        stats = dataset_stats(rows)
        self.assertGreaterEqual(stats["n_docs"], 6)
        self.assertGreaterEqual(stats["n_queries"], 8)
        self.assertIn("keyword", stats["slices"])
        self.assertIn("paraphrase", stats["slices"])
        mapping = map_doc_ids_to_chunks(
            [{"doc_id": "license-alice"}],
            [
                {"id": "c1", "text": "DOCID license-alice. Alice completed her license."},
                {"id": "c2", "text": "unrelated"},
            ],
        )
        self.assertEqual(mapping["license-alice"], {"c1"})

    def test_gold_matches_chunk_or_node_uuid(self):
        from search.evaluate import gold_hit_grades, gold_hit_ids
        from search.metrics import ndcg_at_k

        query = {"gold_doc_ids": ["sku-1"], "gold_grades": {"sku-1": 1.0}}
        mapped = {"sku-1": {"chunk-1"}}
        gold = gold_hit_ids(query, mapped)
        self.assertEqual(gold, {"sku-1"})
        grades = gold_hit_grades(query, mapped)
        self.assertEqual(ndcg_at_k(["sku-1"], gold, 10, grades=grades), 1.0)
        from search.evaluate import canonicalize_hit_ids, invert_doc_chunks

        ranked_chunk = canonicalize_hit_ids(["chunk-1"], invert_doc_chunks(mapped))
        self.assertEqual(ndcg_at_k(ranked_chunk, gold, 10, grades=grades), 1.0)
        self.assertEqual(ndcg_at_k(["other"], gold, 10, grades=grades), 0.0)

    def test_wands_recall_gold_is_exact_and_partial(self):
        from search.evaluate import gold_hit_ids
        from search.metrics import recall_at_k

        query = {
            "gold_doc_ids": ["exact", "partial"],
            "gold_grades": {"exact": 1.0, "partial": 0.5},
        }
        gold = gold_hit_ids(query, {})
        self.assertEqual(gold, {"exact", "partial"})
        self.assertEqual(recall_at_k(["exact", "irr"], gold, 10), 0.5)
        self.assertEqual(recall_at_k(["exact", "partial"], gold, 10), 1.0)
        unlabeled = gold_hit_ids(
            {
                "gold_doc_ids": ["exact"],
                "gold_grades": {"exact": 1.0, "partial": 0.5},
            },
            {},
        )
        self.assertEqual(unlabeled, {"exact", "partial"})
        irr_excluded = gold_hit_ids(
            {
                "gold_doc_ids": ["exact"],
                "gold_grades": {"exact": 1.0, "irr": 0.0},
            },
            {},
        )
        self.assertEqual(irr_excluded, {"exact"})

    def test_replay_fusion_compact_hub_drop(self):
        from search.replay_fusion import replay

        root = Path(__file__).resolve().parents[1]
        graph_path = (
            root / "benchmarks" / "runs" / "search-esci-slice-allcols" / "eval.json"
        )
        passages_path = (
            root / "benchmarks" / "runs" / "search-esci-slice-passages" / "eval.json"
        )
        if not graph_path.exists() or not passages_path.exists():
            self.skipTest("missing paired ESCI slice eval.json files")
        result = replay(
            json.loads(graph_path.read_text(encoding="utf-8")),
            json.loads(passages_path.read_text(encoding="utf-8")),
        )
        self.assertEqual(result["n_queries"], 11)
        self.assertAlmostEqual(result["hub_drop"]["recall@10"], 0.4758, places=3)
        self.assertAlmostEqual(result["hub_drop"]["mrr"], 0.8485, places=3)

    def test_unique_doc_counts_and_channel_lists(self):
        from search.evaluate import channel_id_lists, unique_doc_counts

        raw = ["chunk-1", "sku-1", "chunk-2", "hub:attr:red"]
        mapping = {"chunk-1": "sku-1", "chunk-2": "sku-2"}
        n_raw, n_canon = unique_doc_counts(raw, mapping, k=20)
        self.assertEqual(n_raw, 4)
        self.assertEqual(n_canon, 3)
        lists = channel_id_lists(
            {
                "channel_lists": {
                    "dense": ["chunk-1"],
                    "bm25": ["chunk-2"],
                    "entities": ["sku-1"],
                    "communities": ["sku-3"],
                }
            },
            [{"id": "chunk-1", "channel": "passages", "doc_id": "sku-1"}],
        )
        self.assertEqual(lists["dense_ids"], ["chunk-1"])
        self.assertEqual(lists["entity_ids"], ["sku-1"])
        self.assertEqual(lists["community_ids"], ["sku-3"])
        self.assertEqual(lists["passage_ids"], ["chunk-1"])

    def test_replay_collapse_and_gated_arms(self):
        from search.replay_fusion import pick_offline_winner, replay_offline

        gold = [f"d{i}" for i in range(1, 21)]
        grades = {doc: 1.0 for doc in gold}
        passages_hits = [
            {"id": f"chunk-{item}", "channel": "passages", "doc_id": item}
            for item in gold
        ]
        passages = {
            "queries": [
                {
                    "qid": "q1",
                    "gold_doc_ids": gold,
                    "gold_grades": grades,
                    "hit_ids": gold,
                    "dense_ids": [f"chunk-{item}" for item in gold],
                    "bm25_ids": [f"chunk-{item}" for item in gold],
                    "passage_ids": [f"chunk-{item}" for item in gold],
                    "hits": passages_hits,
                    "metrics": {
                        "ndcg@10": 1.0,
                        "recall@10": 0.5,
                        "recall@20": 1.0,
                        "mrr": 1.0,
                    },
                }
            ]
        }
        fused_raw = [f"chunk-{item}" for item in gold[:8]] + gold[:8] + gold[8:12]
        graph = {
            "queries": [
                {
                    "qid": "q1",
                    "gold_doc_ids": gold,
                    "gold_grades": grades,
                    "hit_ids": gold[:12],
                    "entity_ids": gold[:5],
                    "community_ids": ["d21"],
                    "hits": [
                        {
                            "id": hid,
                            "channel": "passages" if hid.startswith("chunk-") else "entities",
                            "doc_id": hid.replace("chunk-", ""),
                        }
                        for hid in fused_raw
                    ],
                    "metrics": {
                        "ndcg@10": 0.6,
                        "recall@10": 0.4,
                        "recall@20": 0.6,
                        "mrr": 1.0,
                    },
                }
            ]
        }
        result = replay_offline(graph, passages)
        self.assertEqual(result["n_queries"], 1)
        self.assertGreaterEqual(
            result["arms"]["collapse-rrf"]["recall@20"],
            result["arms"]["passages"]["recall@20"] - 1e-9,
        )
        self.assertEqual(result["arms"]["passages"]["unique_docs@20"], 20.0)
        self.assertEqual(result["arms"]["confirmation"]["unique_docs@20"], 20.0)
        self.assertEqual(result["arms"]["expansion-n10"]["unique_docs@20"], 20.0)
        winner = pick_offline_winner(
            {
                "passages": {
                    "ndcg@10": 0.758,
                    "recall@20": 0.847,
                },
                "collapse-rrf": {
                    "ndcg@10": 0.640,
                    "recall@20": 0.847,
                },
                "expansion-n10": {
                    "ndcg@10": 0.758,
                    "recall@20": 0.847,
                },
            }
        )
        self.assertEqual(winner, "expansion-n10")
        self.assertEqual(
            pick_offline_winner(
                {
                    "passages": {"ndcg@10": 0.758, "recall@20": 0.847},
                    "collapse-rrf": {"ndcg@10": 0.640, "recall@20": 0.847},
                    "expansion-n10": {"ndcg@10": 0.758, "recall@20": 0.815},
                }
            ),
            "G08",
        )
        self.assertEqual(
            pick_offline_winner(
                {
                    "passages": {"ndcg@10": 0.758, "recall@20": 0.847},
                    "collapse-rrf": {"ndcg@10": 0.640, "recall@20": 0.847},
                }
            ),
            "G08",
        )

    def test_score_search_dumps_channel_and_doc_id(self):
        from search.client import TimedResult
        from search.evaluate import score_search_result
        from search.metrics import ndcg_at_k, recall_at_k

        result = TimedResult(
            data={
                "hits": [
                    {"id": "chunk-1", "channel": "passages", "node_id": "sku-1"},
                    {"id": "sku-1", "channel": "entities", "node_id": "sku-1"},
                    {"id": "hub:attr:red", "channel": "communities"},
                ],
                "stage_timings": {
                    "stages": [{"stage": "search.retrieve", "wall_ms": 11.0}]
                },
            },
            latency_ms=40.0,
        )
        gold = {"sku-1"}
        grades = {"sku-1": 1.0}
        scored = score_search_result(
            result,
            gold=gold,
            ks=(5, 10, 20, 50),
            grades=grades,
            chunk_to_doc={"chunk-1": "sku-1"},
        )
        self.assertEqual(scored["hit_ids"], ["sku-1", "hub:attr:red"])
        self.assertEqual(
            scored["hits"],
            [
                {
                    "id": "chunk-1",
                    "channel": "passages",
                    "doc_id": "sku-1",
                    "node_id": "sku-1",
                },
                {
                    "id": "sku-1",
                    "channel": "entities",
                    "doc_id": "sku-1",
                    "node_id": "sku-1",
                },
                {
                    "id": "hub:attr:red",
                    "channel": "communities",
                    "doc_id": "hub:attr:red",
                    "node_id": None,
                },
            ],
        )
        self.assertEqual(
            scored["metrics"]["recall@10"],
            recall_at_k(scored["hit_ids"], gold, 10),
        )
        self.assertEqual(
            scored["metrics"]["ndcg@10"],
            ndcg_at_k(scored["hit_ids"], gold, 10, grades=grades),
        )
        self.assertEqual(
            [row["doc_id"] for row in scored["hits"][:1]],
            scored["hit_ids"][:1],
        )
        self.assertEqual(scored["n_unique_docs_raw"], 3)
        self.assertEqual(scored["n_unique_docs_canonical"], 2)
        self.assertEqual(scored["unique_docs_k"], 50)
        self.assertEqual(scored["n_unique_docs_retrieve_canonical"], 2)
        self.assertIn("ndcg@50", scored["metrics"])
        self.assertEqual(scored["entity_ids"], ["sku-1"])
        self.assertEqual(scored["community_ids"], ["hub:attr:red"])
        self.assertEqual(scored["passage_ids"], ["chunk-1"])

    def test_rank_pool_filters_and_keeps_irrelevant_grades(self):
        from search.client import TimedResult
        from search.evaluate import (
            candidate_pool_grades,
            candidate_pool_ids,
            filter_ranked_to_pool,
            score_search_result,
        )
        from search.metrics import ndcg_at_k

        query = {
            "gold_doc_ids": ["p-e"],
            "gold_grades": {"p-e": 1.0, "p-s": 0.1},
            "candidate_doc_ids": ["p-e", "p-s", "p-i"],
            "candidate_grades": {"p-e": 1.0, "p-s": 0.1, "p-i": 0.0},
        }
        self.assertEqual(candidate_pool_ids(query), ["p-e", "p-s", "p-i"])
        self.assertEqual(candidate_pool_grades(query)["p-i"], 0.0)
        self.assertEqual(
            filter_ranked_to_pool(["p-x", "p-e", "p-i", "p-y"], query["candidate_doc_ids"]),
            ["p-e", "p-i"],
        )
        result = TimedResult(
            data={
                "hits": [
                    {"id": "outside", "channel": "passages"},
                    {"id": "p-e", "channel": "passages"},
                    {"id": "p-i", "channel": "passages"},
                ],
                "stage_timings": {
                    "stages": [{"stage": "search.retrieve", "wall_ms": 9.0}]
                },
            },
            latency_ms=12.0,
        )
        scored = score_search_result(
            result,
            gold={"p-e", "p-s"},
            ks=(5, 10, 20),
            grades=candidate_pool_grades(query),
            pool_ids=candidate_pool_ids(query),
        )
        self.assertEqual(scored["hit_ids"], ["p-e", "p-i"])
        expected = ndcg_at_k(
            ["p-e", "p-i"],
            {"p-e", "p-s"},
            20,
            grades=candidate_pool_grades(query),
        )
        self.assertEqual(scored["metrics"]["ndcg@20"], expected)

    def test_rank_pool_ce_uses_injected_predict(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        plugin = root / "plugins" / "search-rerank"
        self.require_local_artifact(plugin / "rerank.py")
        if str(plugin) not in sys.path:
            sys.path.insert(0, str(plugin))
        import rerank as rerank_mod

        from search.rank_pool import run_ce_on_pool

        def predict(pairs):
            scores = []
            for _, text in pairs:
                if "gold" in text:
                    scores.append(2.0)
                elif "partial" in text:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            return scores

        rerank_mod.set_predict(predict)
        try:
            rows = [
                {
                    "type": "doc",
                    "doc_id": "irr",
                    "text": "DOCID irr. Title: other",
                },
                {
                    "type": "doc",
                    "doc_id": "gold",
                    "text": "DOCID gold. Title: gold kettle",
                },
                {
                    "type": "doc",
                    "doc_id": "sub",
                    "text": "DOCID sub. Title: partial kettle",
                },
                {
                    "type": "query",
                    "qid": "esci-1",
                    "query": "kettle",
                    "gold_doc_ids": ["gold", "sub"],
                    "gold_grades": {"gold": 1.0, "sub": 0.1},
                    "candidate_doc_ids": ["irr", "gold", "sub"],
                    "candidate_grades": {"irr": 0.0, "gold": 1.0, "sub": 0.1},
                    "slice": "esci-us",
                },
            ]
            result = run_ce_on_pool(rows, dataset_name="toy-pool.jsonl")
        finally:
            rerank_mod.set_predict(None)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["rank_pool"])
        self.assertTrue(result["rank_pool_ce"])
        self.assertEqual(result["queries"][0]["hit_ids"][0], "gold")
        self.assertGreater(result["metrics"]["ndcg@20"], 0.0)

    def test_ce_on_retrieved_reorders_hits_only(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        plugin = root / "plugins" / "search-rerank"
        self.require_local_artifact(plugin / "rerank.py")
        if str(plugin) not in sys.path:
            sys.path.insert(0, str(plugin))
        import rerank as rerank_mod

        from search.rerank_retrieved import run_ce_on_retrieved

        def predict(pairs):
            scores = []
            for _, text in pairs:
                if "gold" in text:
                    scores.append(2.0)
                elif "partial" in text:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            return scores

        rerank_mod.set_predict(predict)
        try:
            rows = [
                {
                    "type": "doc",
                    "doc_id": "irr",
                    "text": "DOCID irr. Title: other",
                },
                {
                    "type": "doc",
                    "doc_id": "gold",
                    "text": "DOCID gold. Title: gold kettle",
                },
                {
                    "type": "doc",
                    "doc_id": "sub",
                    "text": "DOCID sub. Title: partial kettle",
                },
                {
                    "type": "query",
                    "qid": "esci-1",
                    "query": "kettle",
                    "gold_doc_ids": ["gold", "sub"],
                    "gold_grades": {"gold": 1.0, "sub": 0.1},
                    "slice": "esci-us",
                },
            ]
            eval_result = {
                "k": 50,
                "ks": [5, 10, 20, 50],
                "rank_pool": False,
                "fusion": "rrf",
                "channels": ["passages"],
                "n_docs": 3,
                "queries": [
                    {
                        "qid": "esci-1",
                        "query": "kettle",
                        "slice": "esci-us",
                        "gold_doc_ids": ["gold", "sub"],
                        "gold_grades": {"gold": 1.0, "sub": 0.1},
                        "hit_ids": ["irr", "sub", "gold"],
                        "metrics": {},
                        "retrieve_ms": 12.0,
                        "pool_coverage": None,
                    }
                ],
            }
            result = run_ce_on_retrieved(
                eval_result,
                rows,
                dataset_name="toy-retrieved.jsonl",
            )
        finally:
            rerank_mod.set_predict(None)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["rank_pool_ce"])
        self.assertEqual(result["rerank"], "harness:cross-encoder")
        self.assertEqual(result["queries"][0]["hit_ids"][0], "gold")
        self.assertEqual(set(result["queries"][0]["hit_ids"]), {"irr", "sub", "gold"})
        self.assertGreater(result["metrics"]["ndcg@10"], 0.0)

    def test_electronics_marker_for_esci72_style_titles(self):
        from search.miss import looks_like_electronics

        self.assertTrue(looks_like_electronics("Apple iPad Air 2, 16 GB"))
        self.assertFalse(looks_like_electronics("Price Tags Without String"))

    def test_miss_strata_classifies_head_tail_and_total_miss(self):
        from search.miss_strata import classify_eval, rewrite_query

        self.assertIsNone(rewrite_query("esci-72", "$100 things that are not electronics"))
        self.assertIsNone(rewrite_query("esci-177", "p_num: integer not null i_num: integer"))
        rewritten = rewrite_query(
            "esci-113",
            "&#34;tortillas without interesterified soybean oil”",
        )
        self.assertIsNotNone(rewritten)
        self.assertNotIn("&#34;", rewritten)

        eval_result = {
            "k": 50,
            "queries": [
                {
                    "qid": "q-head",
                    "query": "red mug",
                    "hit_ids": ["g1", "x"] + [f"n{i}" for i in range(48)],
                    "gold_doc_ids": ["g1", "g2"],
                    "gold_grades": {"g1": 1.0, "g2": 0.1},
                    "metrics": {"recall@10": 0.5, "recall@50": 0.5},
                },
                {
                    "qid": "q-tail",
                    "query": "blue mug",
                    "hit_ids": ["n0"] * 10 + ["g2"] + [f"n{i}" for i in range(39)],
                    "gold_doc_ids": ["g2"],
                    "gold_grades": {"g2": 1.0},
                    "metrics": {"recall@10": 0.0, "recall@50": 1.0},
                },
                {
                    "qid": "esci-113",
                    "query": "&#34;tortillas without interesterified soybean oil”",
                    "hit_ids": [f"n{i}" for i in range(50)],
                    "gold_doc_ids": ["g3"],
                    "gold_grades": {"g3": 1.0},
                    "metrics": {"recall@10": 0.0, "recall@50": 0.0},
                },
            ],
        }
        rows = [
            {"type": "doc", "doc_id": "g1", "text": "Title: red mug"},
            {"type": "doc", "doc_id": "g2", "text": "Title: blue mug"},
            {"type": "doc", "doc_id": "g3", "text": "Title: tortillas"},
            {
                "type": "query",
                "qid": "q-head",
                "query": "red mug",
                "gold_doc_ids": ["g1", "g2"],
                "gold_grades": {"g1": 1.0, "g2": 0.1},
            },
            {
                "type": "query",
                "qid": "q-tail",
                "query": "blue mug",
                "gold_doc_ids": ["g2"],
                "gold_grades": {"g2": 1.0},
            },
            {
                "type": "query",
                "qid": "esci-113",
                "query": "&#34;tortillas without interesterified soybean oil”",
                "gold_doc_ids": ["g3"],
                "gold_grades": {"g3": 1.0},
            },
        ]
        taxonomy = classify_eval(eval_result, rows, k=50)
        self.assertEqual(taxonomy["n_queries"], 3)
        self.assertEqual(taxonomy["n_gold"], 4)
        self.assertEqual(taxonomy["stratum_counts"]["head-ok"], 1)
        self.assertEqual(taxonomy["stratum_counts"]["rank-too-low"], 1)
        self.assertEqual(taxonomy["stratum_counts"]["total-miss"], 1)
        self.assertTrue(taxonomy["run_query_side"])
        self.assertEqual(taxonomy["rewritable_qids"], ["esci-113"])

    def test_spell_normalize_nfkc_and_accents(self):
        from search.miss_strata import normalize_spelling, write_spell_jsonl

        self.assertEqual(normalize_spelling("niños!!!"), "ninos")
        self.assertEqual(normalize_spelling("  sofa, azul.  "), "sofa azul")
        self.assertEqual(normalize_spelling("caf\u00e9"), "cafe")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "spell.jsonl"
            summary = write_spell_jsonl(
                [
                    {"type": "doc", "doc_id": "d1", "text": "sofa"},
                    {"type": "query", "qid": "q1", "query": "sofá azul!!"},
                ],
                dest,
            )
            self.assertEqual(summary["n_queries_changed"], 1)
            rows = dest.read_text(encoding="utf-8").strip().splitlines()
            query = json.loads(rows[1])
            self.assertEqual(query["query"], "sofa azul")
            self.assertEqual(query["query_original"], "sofá azul!!")

    def test_doc_meta_keys_copies_generic_catalog_fields(self):
        from search.evaluate import doc_meta_keys

        self.assertEqual(
            doc_meta_keys(
                {
                    "doc_id": "it-bollitore",
                    "brand": "CasaLuce",
                    "color": "argento",
                    "locale": "it",
                    "title": "ignored",
                }
            ),
            {"brand": "CasaLuce", "color": "argento", "locale": "it"},
        )
        self.assertIsNone(doc_meta_keys({"doc_id": "x", "text": "only text"}))

    def test_dense_holdout_excludes_eval_qids(self):
        from search.finetune_esci_ce import held_out_query_ids

        holdout_path = self.require_local_artifact(
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "search_esci_74.jsonl"
        )
        holdout = held_out_query_ids(holdout_path)
        self.assertIn("72", holdout)
        self.assertIn("esci-72", holdout)
        self.assertGreater(len(holdout), 74)

    def test_local_dense_scores_injected_ranks(self):
        from search.local_dense import evaluate_dense

        queries = [
            {
                "qid": "q1",
                "query": "red mug",
                "slice": "toy",
                "gold_doc_ids": ["gold"],
                "gold_grades": {"gold": 1.0, "sub": 0.1},
            }
        ]
        ranked = {"q1": ["gold", "sub", "irr"]}
        metrics, per_query = evaluate_dense(
            ranked, queries, ks=(5, 10, 20, 50), encode_ms=10.0
        )
        self.assertEqual(per_query[0]["hit_ids"][0], "gold")
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertGreater(metrics["ndcg@10"], 0.0)

    def test_exhaustive_ce_ranks_exact_above_irrelevant(self):
        from search.rank_corpus import PROTOCOL, run_exhaustive_ce

        rows = [
            {"type": "doc", "doc_id": "irr", "text": "DOCID irr. Title: other"},
            {"type": "doc", "doc_id": "exact", "text": "DOCID exact. Title: gold kettle"},
            {"type": "doc", "doc_id": "maybe", "text": "DOCID maybe. Title: similar kettle"},
            {
                "type": "query",
                "qid": "esci-toy",
                "query": "kettle",
                "gold_doc_ids": ["exact"],
                "gold_grades": {"exact": 1.0, "maybe": 0.1},
                "slice": "toy",
            },
        ]

        def predict(pairs):
            out = []
            for _, text in pairs:
                if "gold kettle" in text:
                    out.append([0.95, 0.03, 0.01, 0.01])
                elif "similar" in text:
                    out.append([0.05, 0.80, 0.10, 0.05])
                else:
                    out.append([0.01, 0.02, 0.07, 0.90])
            return out

        result = run_exhaustive_ce(
            rows,
            predict=predict,
            model_name="toy-4class",
            dataset_name="toy-corpus.jsonl",
            k=3,
            ks=(5, 10, 20, 50),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["protocol"], PROTOCOL)
        self.assertFalse(result["rank_pool"])
        self.assertEqual(result["queries"][0]["hit_ids"][0], "exact")
        self.assertLess(result["queries"][0]["hit_ids"].index("exact"), result["queries"][0]["hit_ids"].index("irr"))
        self.assertGreater(result["metrics"]["ndcg@10"], 0.0)

    def test_list_overlap_counts_sidecar_unique_golds(self):
        from search.list_overlap import run_union, summarize_overlap, unique_golds

        gold = {"g1", "g2", "g3"}
        extra = unique_golds(
            ["g2", "noise", "g3"],
            ["g1", "g2", "other"],
            gold,
            k=50,
        )
        self.assertEqual(extra, ["g3"])
        passages = {
            "run_id": "passages",
            "queries": [
                {
                    "qid": "q1",
                    "hit_ids": ["g1", "g2", "p1"],
                    "gold_doc_ids": ["g1", "g2", "g3"],
                    "gold_grades": {"g1": 1.0, "g2": 1.0, "g3": 0.1},
                }
            ],
        }
        bge = {
            "run_id": "bge",
            "queries": [{"qid": "q1", "hit_ids": ["g3", "g1", "b1"]}],
        }
        summary = summarize_overlap(passages, {"bge": bge}, k=50)
        self.assertEqual(summary["runs"]["bge"]["unique_gold_hits"], 1)
        rows = [
            {"type": "doc", "doc_id": "g1", "text": "g1"},
            {"type": "doc", "doc_id": "g2", "text": "g2"},
            {"type": "doc", "doc_id": "g3", "text": "g3"},
            {
                "type": "query",
                "qid": "q1",
                "query": "mug",
                "gold_doc_ids": ["g1", "g2", "g3"],
                "gold_grades": {"g1": 1.0, "g2": 1.0, "g3": 0.1},
            },
        ]
        union = run_union(
            [passages, bge],
            rows,
            dataset_name="toy.jsonl",
            k=50,
            ks=(5, 10, 20, 50),
            run_names=["passages", "bge"],
        )
        self.assertEqual(union["channels"], ["harness-union"])
        self.assertIn("g3", union["queries"][0]["hit_ids"][:10])
        self.assertGreaterEqual(union["metrics"]["recall@10"], 1.0)

    def test_cascade_freezes_head_and_injects_tail_golds(self):
        from search.list_overlap import CASCADE_PROTOCOL, cascade_frozen_head, run_cascade

        passages = [f"h{index}" for index in range(10)] + ["tail_gold", "irr"]
        gold = {"h0", "tail_gold", "extra"}
        ranked = cascade_frozen_head(
            passages,
            [["extra", "noise", "h0"]],
            gold,
            head_k=10,
            k=50,
        )
        self.assertEqual(ranked[:10], passages[:10])
        self.assertIn("extra", ranked[10:])
        self.assertNotIn("extra", ranked[:10])
        self.assertIn("tail_gold", ranked)
        rows = [
            {"type": "doc", "doc_id": doc_id, "text": doc_id}
            for doc_id in [*passages, "extra"]
        ]
        rows.append(
            {
                "type": "query",
                "qid": "q1",
                "query": "mug",
                "gold_doc_ids": ["h0", "tail_gold", "extra"],
                "gold_grades": {"h0": 1.0, "tail_gold": 1.0, "extra": 0.1},
            }
        )
        result = run_cascade(
            {"run_id": "passages", "queries": [{"qid": "q1", "hit_ids": passages}]},
            [{"run_id": "bge", "queries": [{"qid": "q1", "hit_ids": ["extra", "noise"]}]}],
            rows,
            dataset_name="toy.jsonl",
            k=50,
            head_k=10,
            ks=(5, 10, 20, 50),
            run_names=["passages", "bge"],
        )
        self.assertEqual(result["protocol"], CASCADE_PROTOCOL)
        self.assertEqual(result["channels"], ["harness-cascade"])
        self.assertEqual(result["queries"][0]["hit_ids"][:10], passages[:10])
        self.assertIn("extra", result["queries"][0]["hit_ids"][10:])
        self.assertEqual(result["metrics"]["recall@10"], 1.0 / 3.0)

    def test_frozen_head_merge_matches_harness_cascade(self):
        from src.core.search.hybrid import frozen_head_merge
        from search.list_overlap import cascade_frozen_head

        passages = [f"h{index}" for index in range(10)] + ["tail_gold", "irr"]
        sidecars = [["extra", "noise", "h0"]]
        gold = {"h0", "tail_gold", "extra"}
        ranked = frozen_head_merge(
            passages,
            sidecars,
            head_k=10,
            k=50,
            prefer_ids=gold,
        )
        self.assertEqual(
            ranked,
            cascade_frozen_head(passages, sidecars, gold, head_k=10, k=50),
        )
        live = frozen_head_merge(passages, sidecars, head_k=10, k=50)
        self.assertEqual(live[:10], passages[:10])
        self.assertIn("extra", live[10:])
        self.assertIn("noise", live[10:])

    def test_cascade_replay_esci74_stored_lists(self):
        from search.dataset import load_records
        from search.list_overlap import load_eval_run, run_cascade

        root = Path(__file__).resolve().parents[1]
        runs = root / "benchmarks" / "runs"
        passages_path = runs / "search-esci-74-passages-k50" / "eval.json"
        bge_path = runs / "search-esci-74-bge-base-k50" / "eval.json"
        colbert_path = runs / "search-esci-74-colbert-k50" / "eval.json"
        dataset = root / "benchmarks" / "data" / "search_esci_74.jsonl"
        if not all(
            path.exists()
            for path in (passages_path, bge_path, colbert_path, dataset)
        ):
            self.skipTest("stored US ESCI n=74 evals not present")
        passages = load_eval_run(passages_path.parent)
        bge = load_eval_run(bge_path.parent)
        colbert = load_eval_run(colbert_path.parent)
        rows = load_records(dataset)
        result = run_cascade(
            passages,
            [bge, colbert],
            rows,
            dataset_name="search_esci_74.jsonl",
            k=50,
            head_k=10,
            ks=(5, 10, 20, 50),
            run_names=["passages", "bge", "colbert"],
        )
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["ndcg@10"], 0.500, places=3)
        self.assertAlmostEqual(metrics["recall@10"], 0.379, places=3)
        self.assertAlmostEqual(metrics["recall@50"], 0.889, places=3)

    def test_ltr_head_promotes_title_overlap_on_held_out_query(self):
        from search.ltr_head import (
            example_from_eval_row,
            features_for_doc,
            fit_pairwise,
            rerank_ids,
            run_ltr_head,
        )

        feats = features_for_doc(
            "navy velvet sofa",
            "hit",
            rrf_ids=["miss", "hit"],
            bm25_ids=["hit"],
            dense_ids=["miss"],
            doc={"title": "Navy velvet sofa two seats", "brand": "AtelierNord"},
        )
        self.assertGreater(feats[3], 0.5)
        self.assertEqual(feats[4], 0.0)
        train_row = {
            "qid": "q-train",
            "query": "navy velvet sofa",
            "hit_ids": ["irr", "gold"],
            "bm25_ids": ["gold"],
            "dense_ids": ["irr"],
            "hits": [
                {"id": "irr", "doc_id": "irr"},
                {"id": "gold", "doc_id": "gold"},
            ],
            "gold_doc_ids": ["gold"],
            "gold_grades": {"gold": 1.0},
        }
        test_row = {
            "qid": "q-test",
            "query": "navy velvet sofa",
            "hit_ids": ["irr", "gold"],
            "bm25_ids": ["gold"],
            "dense_ids": ["irr"],
            "hits": [
                {"id": "irr", "doc_id": "irr"},
                {"id": "gold", "doc_id": "gold"},
            ],
            "gold_doc_ids": ["gold"],
            "gold_grades": {"gold": 1.0},
        }
        docs = {
            "gold": {"doc_id": "gold", "title": "Navy velvet sofa"},
            "irr": {"doc_id": "irr", "title": "plastic kettle lid"},
        }
        train_ex = example_from_eval_row(train_row, docs, k=10)
        test_ex = example_from_eval_row(test_row, docs, k=10)
        self.assertIsNotNone(train_ex)
        self.assertIsNotNone(test_ex)
        weights = fit_pairwise([train_ex], epochs=20, seed=0)
        ranked = rerank_ids(test_ex, weights)
        self.assertEqual(ranked[0], "gold")
        result = run_ltr_head(
            {"run_id": "toy", "queries": [train_row, test_row]},
            [
                {"type": "doc", "doc_id": "gold", "title": "Navy velvet sofa", "text": "Title: Navy velvet sofa"},
                {"type": "doc", "doc_id": "irr", "title": "plastic kettle lid", "text": "Title: plastic kettle lid"},
                {
                    "type": "query",
                    "qid": "q-train",
                    "query": "navy velvet sofa",
                    "gold_doc_ids": ["gold"],
                    "gold_grades": {"gold": 1.0},
                },
                {
                    "type": "query",
                    "qid": "q-test",
                    "query": "navy velvet sofa",
                    "gold_doc_ids": ["gold"],
                    "gold_grades": {"gold": 1.0},
                },
            ],
            dataset_name="toy.jsonl",
            k=10,
            ks=(5, 10),
            n_folds=2,
        )
        self.assertEqual(result["protocol"], "ltr-head-cv")
        self.assertGreaterEqual(result["metrics"]["recall@10"], 1.0)

    def test_ltr_pair_policy_skips_unlabeled_for_other_query_neg(self):
        from search.ltr_head import (
            PAIR_OTHER_QUERY_NEG,
            PAIR_UNLABELED_ZERO,
            collect_pairs,
            example_from_eval_row,
            features_for_doc,
            other_query_gold_ids,
        )

        row_a = {
            "qid": "q-a",
            "query": "navy sofa",
            "hit_ids": ["gold-a", "gold-b", "unlabeled"],
            "hits": [
                {"id": "gold-a", "doc_id": "gold-a"},
                {"id": "gold-b", "doc_id": "gold-b"},
                {"id": "unlabeled", "doc_id": "unlabeled"},
            ],
            "gold_doc_ids": ["gold-a"],
            "gold_grades": {"gold-a": 1.0},
        }
        row_b = {
            "qid": "q-b",
            "query": "steel kettle",
            "hit_ids": ["gold-b", "gold-a", "unlabeled"],
            "hits": [
                {"id": "gold-b", "doc_id": "gold-b"},
                {"id": "gold-a", "doc_id": "gold-a"},
                {"id": "unlabeled", "doc_id": "unlabeled"},
            ],
            "gold_doc_ids": ["gold-b"],
            "gold_grades": {"gold-b": 1.0},
        }
        docs = {
            "gold-a": {"doc_id": "gold-a", "title": "Navy sofa"},
            "gold-b": {"doc_id": "gold-b", "title": "Steel kettle"},
            "unlabeled": {"doc_id": "unlabeled", "title": "random cable"},
        }
        ex_a = example_from_eval_row(row_a, docs, k=10)
        ex_b = example_from_eval_row(row_b, docs, k=10)
        other = other_query_gold_ids([ex_a, ex_b], "q-a")
        self.assertEqual(other, {"gold-b"})
        zero_pairs = collect_pairs(ex_a, pair_policy=PAIR_UNLABELED_ZERO)
        neg_pairs = collect_pairs(
            ex_a, pair_policy=PAIR_OTHER_QUERY_NEG, other_gold=other
        )
        zero_docs = {(ex_a["ids"][hi], ex_a["ids"][lo]) for hi, lo in zero_pairs}
        neg_docs = {(ex_a["ids"][hi], ex_a["ids"][lo]) for hi, lo in neg_pairs}
        self.assertIn(("gold-a", "unlabeled"), zero_docs)
        self.assertNotIn(("gold-a", "unlabeled"), neg_docs)
        self.assertIn(("gold-a", "gold-b"), neg_docs)
        feats = features_for_doc(
            "navy sofa",
            "gold-a",
            rrf_ids=["gold-a"],
            bm25_ids=["gold-a"],
            dense_ids=["gold-a"],
            doc={"title": "Navy sofa"},
            ce_gain=0.42,
        )
        self.assertEqual(len(feats), 7)
        self.assertAlmostEqual(float(feats[-1]), 0.42)
        from search.ltr_head import rank_train_group

        grouped = rank_train_group(
            ex_a, pair_policy=PAIR_OTHER_QUERY_NEG, other_gold=other
        )
        self.assertIsNotNone(grouped)
        _, labels = grouped
        self.assertEqual(len(labels), 2)

    def test_ltr_lightgbm_promotes_high_ce_gain_on_held_out_query(self):
        from search.ltr_head import (
            PAIR_OTHER_QUERY_NEG,
            example_from_eval_row,
            fit_lightgbm,
            rerank_ids_model,
        )

        n_queries = 8
        docs: dict[str, dict[str, str]] = {}
        rows: list[dict] = []
        for index in range(n_queries):
            gold = f"gold-{index}"
            other = f"gold-{(index + 1) % n_queries}"
            miss = f"miss-{index}"
            docs[gold] = {"doc_id": gold, "title": f"Navy sofa {index}"}
            docs[miss] = {"doc_id": miss, "title": f"plastic lid {index}"}
            rows.append(
                {
                    "qid": f"q-{index}",
                    "query": "navy sofa",
                    "hit_ids": [miss, other, gold],
                    "bm25_ids": [gold],
                    "dense_ids": [miss],
                    "hits": [
                        {"id": miss, "doc_id": miss},
                        {"id": other, "doc_id": other},
                        {"id": gold, "doc_id": gold},
                    ],
                    "gold_doc_ids": [gold],
                    "gold_grades": {gold: 1.0},
                    "ce": {gold: 0.95, other: 0.05, miss: 0.01},
                }
            )
        examples = [
            example_from_eval_row(
                row, docs, k=10, ce_scores=row["ce"]
            )
            for row in rows
        ]
        self.assertTrue(all(item is not None for item in examples))
        model = fit_lightgbm(
            examples[:-1],
            pair_policy=PAIR_OTHER_QUERY_NEG,
            min_data_in_leaf=1,
        )
        ranked = rerank_ids_model(examples[-1], model)
        self.assertEqual(ranked[0], "gold-7")
        self.assertGreater(float(sum(model.feature_importances_)), 0.0)

    def test_ltr_apply_trains_on_other_run_not_eval_qid(self):
        from search.ltr_head import PROTOCOL_APPLY, run_ltr_head

        train_docs = [
            {"type": "doc", "doc_id": "gold-t", "title": "Navy velvet sofa", "text": "Title: Navy velvet sofa"},
            {"type": "doc", "doc_id": "irr-t", "title": "plastic lid", "text": "Title: plastic lid"},
        ]
        eval_docs = [
            {"type": "doc", "doc_id": "gold-e", "title": "Navy velvet sofa", "text": "Title: Navy velvet sofa"},
            {"type": "doc", "doc_id": "irr-e", "title": "plastic lid", "text": "Title: plastic lid"},
        ]
        train_queries = [
            {
                "type": "query",
                "qid": "q-train",
                "query": "navy velvet sofa",
                "gold_doc_ids": ["gold-t"],
                "gold_grades": {"gold-t": 1.0},
            }
        ]
        eval_queries = [
            {
                "type": "query",
                "qid": "q-eval",
                "query": "navy velvet sofa",
                "gold_doc_ids": ["gold-e"],
                "gold_grades": {"gold-e": 1.0},
            }
        ]
        train_row = {
            "qid": "q-train",
            "query": "navy velvet sofa",
            "hit_ids": ["irr-t", "gold-t"],
            "bm25_ids": ["gold-t"],
            "dense_ids": ["irr-t"],
            "hits": [
                {"id": "irr-t", "doc_id": "irr-t"},
                {"id": "gold-t", "doc_id": "gold-t"},
            ],
            "gold_doc_ids": ["gold-t"],
            "gold_grades": {"gold-t": 1.0},
        }
        eval_row = {
            "qid": "q-eval",
            "query": "navy velvet sofa",
            "hit_ids": ["irr-e", "gold-e"],
            "bm25_ids": ["gold-e"],
            "dense_ids": ["irr-e"],
            "hits": [
                {"id": "irr-e", "doc_id": "irr-e"},
                {"id": "gold-e", "doc_id": "gold-e"},
            ],
            "gold_doc_ids": ["gold-e"],
            "gold_grades": {"gold-e": 1.0},
        }
        result = run_ltr_head(
            {"run_id": "toy-eval", "queries": [eval_row]},
            eval_docs + eval_queries,
            dataset_name="toy-eval.jsonl",
            k=10,
            ks=(5, 10),
            n_folds=2,
            train_eval_result={"run_id": "toy-train", "queries": [train_row]},
            train_rows=train_docs + train_queries,
            train_source_run="toy-train",
        )
        self.assertEqual(result["protocol"], PROTOCOL_APPLY)
        self.assertEqual(result["ltr_train_run"], "toy-train")
        self.assertEqual(result["ltr_n_train_queries"], 1)
        self.assertEqual(result["queries"][0]["qid"], "q-eval")
        self.assertEqual(result["queries"][0]["hit_ids"][0], "gold-e")
        self.assertGreaterEqual(result["metrics"]["recall@10"], 1.0)

    def test_ltr_apply_lightgbm_trains_on_other_run_not_eval_qid(self):
        from search.ltr_head import HEAD_LIGHTGBM, PROTOCOL_APPLY, run_ltr_head

        train_docs: list[dict] = []
        train_queries: list[dict] = []
        train_eval_rows: list[dict] = []
        for index in range(8):
            gold = f"gold-t-{index}"
            miss = f"miss-t-{index}"
            train_docs.extend(
                [
                    {
                        "type": "doc",
                        "doc_id": gold,
                        "title": f"Navy velvet sofa {index}",
                        "text": f"Title: Navy velvet sofa {index}",
                    },
                    {
                        "type": "doc",
                        "doc_id": miss,
                        "title": f"plastic lid {index}",
                        "text": f"Title: plastic lid {index}",
                    },
                ]
            )
            qid = f"q-train-{index}"
            train_queries.append(
                {
                    "type": "query",
                    "qid": qid,
                    "query": "navy velvet sofa",
                    "gold_doc_ids": [gold],
                    "gold_grades": {gold: 1.0},
                }
            )
            train_eval_rows.append(
                {
                    "qid": qid,
                    "query": "navy velvet sofa",
                    "hit_ids": [miss, gold],
                    "bm25_ids": [gold],
                    "dense_ids": [miss],
                    "hits": [
                        {"id": miss, "doc_id": miss},
                        {"id": gold, "doc_id": gold},
                    ],
                    "gold_doc_ids": [gold],
                    "gold_grades": {gold: 1.0},
                }
            )
        eval_docs = [
            {
                "type": "doc",
                "doc_id": "gold-e",
                "title": "Navy velvet sofa",
                "text": "Title: Navy velvet sofa",
            },
            {
                "type": "doc",
                "doc_id": "irr-e",
                "title": "plastic lid",
                "text": "Title: plastic lid",
            },
        ]
        eval_queries = [
            {
                "type": "query",
                "qid": "q-eval",
                "query": "navy velvet sofa",
                "gold_doc_ids": ["gold-e"],
                "gold_grades": {"gold-e": 1.0},
            }
        ]
        eval_row = {
            "qid": "q-eval",
            "query": "navy velvet sofa",
            "hit_ids": ["irr-e", "gold-e"],
            "bm25_ids": ["gold-e"],
            "dense_ids": ["irr-e"],
            "hits": [
                {"id": "irr-e", "doc_id": "irr-e"},
                {"id": "gold-e", "doc_id": "gold-e"},
            ],
            "gold_doc_ids": ["gold-e"],
            "gold_grades": {"gold-e": 1.0},
        }
        result = run_ltr_head(
            {"run_id": "toy-eval", "queries": [eval_row]},
            eval_docs + eval_queries,
            dataset_name="toy-eval.jsonl",
            k=10,
            ks=(5, 10),
            n_folds=2,
            ltr_head=HEAD_LIGHTGBM,
            train_eval_result={"run_id": "toy-train", "queries": train_eval_rows},
            train_rows=train_docs + train_queries,
            train_source_run="toy-train",
        )
        self.assertEqual(result["protocol"], PROTOCOL_APPLY)
        self.assertEqual(result["ltr_train_run"], "toy-train")
        self.assertEqual(result["ltr_n_train_queries"], 8)
        self.assertEqual(result["rerank"], "ltr-lightgbm")
        self.assertEqual(result["queries"][0]["qid"], "q-eval")
        self.assertNotEqual(result["queries"][0]["qid"], "q-train-0")

    def test_export_hybrid_lists_holdout_unlabeled_is_i(self):
        from search.export_hybrid_lists import (
            PROTECTED_OUT_NAMES,
            export_hybrid_lists,
            rows_from_eval,
        )

        self.assertIn("esci_retrieved_lists.jsonl", PROTECTED_OUT_NAMES)
        eval_result = {
            "queries": [
                {
                    "qid": "esci-hold",
                    "query": "navy sofa",
                    "hit_ids": ["gold-h", "miss-h"],
                    "hits": [
                        {"id": "gold-h", "doc_id": "gold-h"},
                        {"id": "miss-h", "doc_id": "miss-h"},
                    ],
                    "gold_grades": {"gold-h": 1.0},
                },
                {
                    "qid": "esci-ok",
                    "query": "navy sofa",
                    "hit_ids": ["gold-ok", "miss-ok"],
                    "hits": [
                        {"id": "gold-ok", "doc_id": "gold-ok"},
                        {"id": "miss-ok", "doc_id": "miss-ok"},
                    ],
                    "gold_grades": {"gold-ok": 1.0},
                },
            ]
        }
        docs = [
            {
                "type": "doc",
                "doc_id": "gold-ok",
                "title": "Navy sofa",
                "text": "Title: Navy sofa",
            },
            {
                "type": "doc",
                "doc_id": "miss-ok",
                "title": "plastic lid",
                "text": "Title: plastic lid",
            },
        ]
        queries = [
            {
                "type": "query",
                "qid": "esci-ok",
                "query": "navy sofa",
                "gold_doc_ids": ["gold-ok"],
                "gold_grades": {"gold-ok": 1.0},
            },
            {
                "type": "query",
                "qid": "esci-hold",
                "query": "navy sofa",
                "gold_doc_ids": ["gold-h"],
                "gold_grades": {"gold-h": 1.0},
            },
        ]
        rows = rows_from_eval(
            eval_result,
            docs=docs,
            queries=queries,
            holdout={"esci-hold", "hold"},
            k=50,
        )
        qids = {row["query_id"] for row in rows}
        self.assertNotIn("esci-hold", qids)
        self.assertIn("esci-ok", qids)
        by_pid = {row["product_id"]: row["label"] for row in rows}
        self.assertEqual(by_pid["gold-ok"], "E")
        self.assertEqual(by_pid["miss-ok"], "I")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            holdout_path = root / "holdout.jsonl"
            out = root / "hybrid.jsonl"
            dataset.write_text(
                json.dumps(docs[0])
                + "\n"
                + json.dumps(docs[1])
                + "\n"
                + json.dumps(queries[0])
                + "\n",
                encoding="utf-8",
            )
            holdout_path.write_text(
                json.dumps(
                    {
                        "type": "query",
                        "qid": "esci-hold",
                        "query": "navy sofa",
                        "gold_doc_ids": ["gold-h"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            meta = export_hybrid_lists(
                eval_result=eval_result,
                dataset_path=dataset,
                holdout_path=holdout_path,
                out_path=out,
                k=50,
            )
            self.assertEqual(meta["source"], "hybrid-k50")
            self.assertEqual(meta["n_queries"], 1)
            with self.assertRaises(ValueError):
                export_hybrid_lists(
                    eval_result=eval_result,
                    dataset_path=dataset,
                    holdout_path=holdout_path,
                    out_path=root / "esci_retrieved_lists.jsonl",
                    k=50,
                )

    def test_retrieved_lists_unlabeled_is_irrelevant(self):
        from search.finetune_esci_4class import held_out_query_ids
        from search.mine_retrieved_lists import (
            is_held_out,
            labeled_hits,
            mine_from_groups,
            select_groups,
        )
        from search.pool_first_stage import Bm25Index, rank_docs, tokenize

        docs = [
            ("B001", tokenize("red ceramic mug")),
            ("B002", tokenize("blue dinner plate")),
            ("B003", tokenize("red mug gift set")),
        ]
        index = Bm25Index(docs)
        qrels = {"B001": "E", "B003": "S"}
        scores = index.scores(tokenize("red ceramic mug"))
        ranked = rank_docs(scores, ["B001", "B002", "B003"])
        hits = labeled_hits(ranked, qrels, k=3)
        by_id = dict(hits)
        self.assertEqual(by_id["B001"], "E")
        self.assertEqual(by_id["B003"], "S")
        self.assertEqual(by_id["B002"], "I")
        self.assertTrue(any(label == "I" for _, label in hits))

        groups = {
            "q1": {
                "query": "red ceramic mug",
                "qrels": {"B001": "E", "B003": "S"},
            }
        }
        passages = {
            "B001": "Title: red ceramic mug",
            "B002": "Title: blue dinner plate",
            "B003": "Title: red mug gift set",
        }
        rows = mine_from_groups(groups, passages, k=3)
        labels = {row["product_id"]: row["label"] for row in rows}
        self.assertEqual(labels["B001"], "E")
        self.assertEqual(labels["B002"], "I")

        holdout_path = self.require_local_artifact(
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "search_esci_74.jsonl"
        )
        holdout = held_out_query_ids(holdout_path)
        self.assertTrue(is_held_out("72", holdout))
        selected = select_groups(
            {
                "72": {"query": "not electronics", "qrels": {"x": "E"}},
                "esci-113": {"query": "html", "qrels": {"y": "E"}},
                "900001": {"query": "ok", "qrels": {"z": "E"}},
            },
            holdout=holdout,
            max_queries=10,
            seed=1,
        )
        self.assertNotIn("72", selected)
        self.assertNotIn("esci-113", selected)
        self.assertIn("900001", selected)

    def test_local_colbert_fake_encoder_ranks_match(self):
        from search.local_colbert import PLUGIN_DIR, run_local_colbert

        import sys

        self.require_local_artifact(PLUGIN_DIR / "encode.py")
        if str(PLUGIN_DIR) not in sys.path:
            sys.path.insert(0, str(PLUGIN_DIR))
        import encode as colbert_encode

        def fake_tokens(text: str):
            blob = (text or "").lower()
            if "match" in blob or blob == "q":
                return [[1.0, 0.0]]
            return [[0.0, 1.0]]

        colbert_encode.set_encoder(fake_tokens)
        try:
            rows = [
                {"type": "doc", "doc_id": "hit", "text": "match"},
                {"type": "doc", "doc_id": "miss", "text": "other"},
                {
                    "type": "query",
                    "qid": "q1",
                    "query": "q",
                    "gold_doc_ids": ["hit"],
                    "gold_grades": {"hit": 1.0},
                },
            ]
            result = run_local_colbert(
                rows,
                dataset_name="toy",
                k=2,
                ks=(5, 10, 20, 50),
                brain_id="harness-local-colbert",
            )
            self.assertEqual(result["channels"], ["harness-colbert"])
            self.assertEqual(result["queries"][0]["hit_ids"][0], "hit")
        finally:
            colbert_encode.set_encoder(None)

    def test_failed_report_skips_ledger(self):
        from search.report import entry_from_report, update_reports_json

        failed = {
            "status": "failed",
            "run_id": "search-fail",
            "brain_id": "searchbenchsmoke",
            "dataset": "search_toy.jsonl",
            "n_queries": 8,
            "ndcg@10": 0.9,
        }
        self.assertIsNone(entry_from_report(failed))
        empty_ok = {
            "status": "ok",
            "run_id": "search-empty",
            "n_queries": 0,
        }
        self.assertIsNone(entry_from_report(empty_ok))

        ledger = {
            "schema_version": 2,
            "benchmarks": {
                "locomo": {"name": "LoCoMo", "leaderboard": [{"run_id": "keep-me"}]},
                "recsys": {"name": "RecSys", "leaderboard": [{"run_id": "keep-recsys"}]},
                "search": {"name": "Search (hybrid BM25 + dense)", "leaderboard": []},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "REPORTS.json"
            path.write_text(json.dumps(ledger), encoding="utf-8")
            with patch("search.report.REPORTS_PATH", path):
                update_reports_json(failed)
            after = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                after["benchmarks"]["locomo"]["leaderboard"][0]["run_id"],
                "keep-me",
            )
            self.assertEqual(
                after["benchmarks"]["recsys"]["leaderboard"][0]["run_id"],
                "keep-recsys",
            )
            self.assertEqual(after["benchmarks"]["search"]["leaderboard"], [])

            ok = {
                "status": "ok",
                "run_id": "search-ok",
                "brain_id": "searchbenchsmoke",
                "dataset": "search_toy.jsonl",
                "fusion": "rrf",
                "n_queries": 8,
                "ndcg@10": 0.5,
                "recall@10": 0.6,
                "mrr": 0.4,
                "p50_retrieve_ms": 12.0,
                "p95_retrieve_ms": 30.0,
                "git_sha": "abc",
                "recorded_at": "2026-08-18T00:00:00+00:00",
            }
            with patch("search.report.REPORTS_PATH", path):
                update_reports_json(ok)
            after_ok = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                after_ok["benchmarks"]["locomo"]["leaderboard"][0]["run_id"],
                "keep-me",
            )
            self.assertEqual(len(after_ok["benchmarks"]["search"]["leaderboard"]), 1)
            self.assertEqual(
                after_ok["benchmarks"]["search"]["leaderboard"][0]["run_id"],
                "search-ok",
            )

    def test_cli_refuses_memory_brain(self):
        from search.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--brain", "locomoconv26", "dataset-stats"])
        self.assertIn("Refusing brain_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
