import unittest
from types import SimpleNamespace

from plugins_path_helper import load_export_edges


class FakePage:
    def __init__(self, results, total):
        self.results = results
        self.total = total


class FakeGraph:
    def __init__(self, users, hubs):
        self.users = users
        self.hubs = hubs

    def search_entities(self, brain_id="default", limit=10, skip=0, node_labels=None, query_text=None):
        assert node_labels == ["USER"]
        chunk = self.users[skip : skip + limit]
        return FakePage(chunk, len(self.users))

    def get_event_centric_neighbors(self, nodes, brain_id="default"):
        return self.hubs


class ExportEdgesTests(unittest.TestCase):
    def test_export_two_hubs(self):
        export_edges = load_export_edges()
        u1 = SimpleNamespace(uuid="user:u1", name="u1", labels=["USER"])
        u2 = SimpleNamespace(uuid="user:u2", name="u2", labels=["USER"])
        e1 = SimpleNamespace(uuid="evt:1", name="Purchase", labels=["EVENT"])
        e2 = SimpleNamespace(uuid="evt:2", name="View", labels=["EVENT"])
        p1 = SimpleNamespace(uuid="item:sku-1", name="sku-1", labels=["PRODUCT"])
        p2 = SimpleNamespace(uuid="item:sku-2", name="sku-2", labels=["PRODUCT"])
        pred = SimpleNamespace(name="MADE", uuid="r1")
        pred2 = SimpleNamespace(name="TARGETED", uuid="r2")
        hubs = [
            (u1, pred, e1, pred2, p1),
            (u2, pred, e2, pred2, p2),
        ]
        graph = FakeGraph([u1, u2], hubs)
        edges = export_edges.export_user_item_edges(graph, "demorecsys")
        self.assertEqual(len(edges), 2)
        by_user = {e["user_id"]: e for e in edges}
        self.assertEqual(by_user["u1"]["item_id"], "sku-1")
        self.assertEqual(by_user["u1"]["weight"], 3.0)
        self.assertEqual(by_user["u2"]["weight"], 1.0)

    def test_refuse_forbidden_brain(self):
        export_edges = load_export_edges()
        with self.assertRaises(ValueError):
            export_edges.export_user_item_edges(FakeGraph([], []), "beam1m1clean")


if __name__ == "__main__":
    unittest.main()
