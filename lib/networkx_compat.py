"""Compatibility helpers for NetworkX node-link JSON files."""

import inspect

from networkx.readwrite import json_graph


def node_link_graph(data):
    """Load node-link JSON written with either ``links`` or ``edges``.

    NetworkX 3.6 changed the default edge-list key from ``links`` to
    ``edges``.  The LatTE-compatible datasets used by the MLU branch still
    contain ``links``.  Older NetworkX releases use the ``link`` keyword,
    while newer releases use ``edges``.
    """

    if "edges" in data:
        edge_key = "edges"
    elif "links" in data:
        edge_key = "links"
    else:
        return json_graph.node_link_graph(data)

    parameters = inspect.signature(json_graph.node_link_graph).parameters
    if "edges" in parameters:
        return json_graph.node_link_graph(data, edges=edge_key)
    if "link" in parameters:
        return json_graph.node_link_graph(data, link=edge_key)
    return json_graph.node_link_graph(data)
