import tempfile
import unittest
from pathlib import Path


class LightGCNTrainSmokeTests(unittest.TestCase):
    def test_train_and_infer_toy_edges(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        plugin = root / "plugins" / "recsys-gnn"
        if not (plugin / "models" / "lightgcn.py").is_file():
            self.skipTest("optional recsys-gnn plugin is not installed")
        if str(plugin) not in sys.path:
            sys.path.insert(0, str(plugin))

        from models.lightgcn import train_and_save
        from models.artifacts import load_artifacts
        from models.infer import rank_items

        edges = []
        for u in range(5):
            for i in range(3):
                edges.append(
                    {
                        "user_id": f"u{u}",
                        "item_id": f"sku-{i}",
                        "user_uuid": f"user:u{u}",
                        "item_uuid": f"item:sku-{i}",
                        "user_name": f"u{u}",
                        "item_name": f"sku-{i}",
                        "weight": 1.0,
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            meta = train_and_save(
                edges,
                brain_id="demorecsys",
                epochs=3,
                embedding_dim=8,
                n_layers=2,
                out_dir=Path(tmp),
            )
            self.assertEqual(meta["n_users"], 5)
            self.assertEqual(meta["n_items"], 3)
            arts = load_artifacts("demorecsys", base=Path(tmp))
            self.assertIsNotNone(arts)
            self.assertEqual(arts.user_emb.shape, (5, 8))
            self.assertEqual(arts.item_emb.shape, (3, 8))
            out = rank_items(arts, "u0", top_k=2, exclude_seen=False)
            self.assertEqual(out["status"], "ok")
            self.assertEqual(len(out["items"]), 2)


if __name__ == "__main__":
    unittest.main()
