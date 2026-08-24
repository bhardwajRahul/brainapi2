import unittest
from pathlib import Path


class SearchEsci4ClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        root = Path(__file__).resolve().parents[1]
        bench = root / "benchmarks"
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))

    def test_weighted_scores_match_esci_gains(self):
        from search.finetune_esci_4class import weighted_scores

        exact = weighted_scores([[1.0, 0.0, 0.0, 0.0]])
        sub = weighted_scores([[0.0, 1.0, 0.0, 0.0]])
        comp = weighted_scores([[0.0, 0.0, 1.0, 0.0]])
        irr = weighted_scores([[0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(exact, [1.0])
        self.assertEqual(sub, [0.1])
        self.assertEqual(comp, [0.01])
        self.assertEqual(irr, [0.0])
        mixed = weighted_scores([[0.5, 0.5, 0.0, 0.0]])
        self.assertAlmostEqual(mixed[0], 0.55)

    def test_rank_doc_ids_orders_by_weighted_score(self):
        from search.finetune_esci_4class import rank_doc_ids, weighted_scores

        doc_ids = ["irr", "exact", "sub"]
        scores = weighted_scores(
            [
                [0.05, 0.05, 0.1, 0.8],
                [0.9, 0.05, 0.05, 0.0],
                [0.1, 0.8, 0.05, 0.05],
            ]
        )
        self.assertEqual(rank_doc_ids(doc_ids, scores), ["exact", "sub", "irr"])

    def test_class_weights_upweight_rare_labels(self):
        from search.finetune_esci_4class import class_weights

        weights = class_weights([80, 10, 5, 5])
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[3], weights[0])

    def test_held_out_query_ids_strip_esci_prefix(self):
        import json
        import tempfile

        from search.finetune_esci_4class import held_out_query_ids

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slice.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "doc",
                        "doc_id": "p1",
                        "text": "DOCID p1.\nTitle: kettle",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "query",
                        "qid": "esci-99",
                        "query": "kettle",
                        "gold_doc_ids": ["p1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            holdout = held_out_query_ids(path)
        self.assertIn("esci-99", holdout)
        self.assertIn("99", holdout)

    def test_run_4class_on_pool_keeps_irrelevant_and_ranks_exact_first(self):
        from search.rank_pool_4class import run_4class_on_pool

        def predict(pairs):
            scores = []
            for _, text in pairs:
                if "gold kettle" in text:
                    scores.append([0.85, 0.1, 0.05, 0.0])
                elif "partial kettle" in text:
                    scores.append([0.1, 0.8, 0.05, 0.05])
                else:
                    scores.append([0.05, 0.05, 0.1, 0.8])
            return scores

        rows = [
            {"type": "doc", "doc_id": "irr", "text": "DOCID irr. Title: lamp"},
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
        result = run_4class_on_pool(
            rows,
            dataset_name="toy.jsonl",
            predict=predict,
            model_name="toy-4class",
            brain_id="searchbenchesci74",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["n_queries"], 1)
        self.assertTrue(result["rank_pool"])
        ranked = result["queries"][0]["hit_ids"]
        self.assertEqual(ranked, ["gold", "sub", "irr"])
        self.assertEqual(ranked[-1], "irr")
        self.assertGreater(result["metrics"]["ndcg@20"], 0.9)

    def test_persist_tokenizer_max_length_writes_train_length(self):
        import tempfile
        from types import SimpleNamespace

        from search.finetune_esci_4class import persist_tokenizer_max_length

        class FakeTokenizer:
            def __init__(self):
                self.model_max_length = 10**18
                self.saved = None

            def save_pretrained(self, path):
                self.saved = path

        tokenizer = FakeTokenizer()
        model = SimpleNamespace(max_length=192, tokenizer=tokenizer)
        with tempfile.TemporaryDirectory() as tmp:
            persist_tokenizer_max_length(model, Path(tmp))
            self.assertEqual(tokenizer.model_max_length, 192)
            self.assertEqual(tokenizer.saved, tmp)


if __name__ == "__main__":
    unittest.main()
