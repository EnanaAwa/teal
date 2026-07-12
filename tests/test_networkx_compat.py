import unittest
from unittest import mock

from lib.networkx_compat import node_link_graph


class NodeLinkGraphCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.nodes = [{"id": 0}, {"id": 1}]
        self.edge = {"source": 0, "target": 1}

    def test_loads_links_payload(self):
        graph = node_link_graph({"nodes": self.nodes, "links": [self.edge]})
        self.assertEqual(list(graph.edges()), [(0, 1)])

    def test_loads_edges_payload(self):
        graph = node_link_graph({"nodes": self.nodes, "edges": [self.edge]})
        self.assertEqual(list(graph.edges()), [(0, 1)])

    def test_uses_legacy_link_keyword_when_required(self):
        def legacy_loader(data, link="links"):
            return data[link]

        with mock.patch(
            "lib.networkx_compat.json_graph.node_link_graph",
            new=legacy_loader,
        ):
            self.assertEqual(
                node_link_graph({"nodes": self.nodes, "edges": [self.edge]}),
                [self.edge],
            )

    def test_does_not_mask_loader_type_errors(self):
        def broken_loader(data, *, edges="edges"):
            raise TypeError("malformed node-link payload")

        with mock.patch(
            "lib.networkx_compat.json_graph.node_link_graph",
            new=broken_loader,
        ):
            with self.assertRaisesRegex(TypeError, "malformed node-link payload"):
                node_link_graph({"nodes": self.nodes, "links": [self.edge]})


if __name__ == "__main__":
    unittest.main()
