import tempfile
import unittest
from pathlib import Path


class MovieLensNormalizeTests(unittest.TestCase):
    def test_filter_and_stats(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        bench = root / "benchmarks"
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))

        from recsys.dataset import dataset_stats, filter_interactions, write_jsonl
        from recsys.evaluate import load_interactions

        rows = []
        for u in range(5):
            for i in range(u + 2):  # users 0..4 have 2..6 interactions
                rows.append(
                    {
                        "user_id": f"mlu{u}",
                        "item_id": f"mlm{i}",
                        "behavior": "purchase",
                        "timestamp": f"2024-01-0{i+1}T00:00:00Z",
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(rows, Path(tmp) / "x.jsonl")
            loaded = load_interactions(path)
            self.assertEqual(len(loaded), len(rows))
            filtered = filter_interactions(
                loaded, min_interactions=4, max_users=10
            )
            # users 2,3,4 have 4,5,6 ix
            self.assertEqual(len({r["user_id"] for r in filtered}), 3)
            stats = dataset_stats(filtered)
            self.assertEqual(stats["n_users"], 3)


if __name__ == "__main__":
    unittest.main()
