import numpy as np
import networkx as nx

import torch
import json
import pickle
import os

from collections import defaultdict
from typing import (
    List,
    Optional
)

class SnapshotReader(object):
    def __init__(
        self,
        dataset_dir: str,
        topology: str,
        topology_name: str,
        pairs_name: str,
        tm_name: str,
        num_paths_per_pair: int,
        failure_id: Optional[int] = None,
    ):
        self.dataset_dir = dataset_dir
        self.topology = topology
        
        # FIXME: failure id
        #self.failure_id = args.failure_id
        self.failure_id = failure_id
        self.num_paths_per_pair = num_paths_per_pair

        topo_path = os.path.join(
            dataset_dir, "Topology", topology_name
        )
        (
            self.graph, self.link_caps, self.link_caps_normed
        ) = _read_graph_from_json(
            topology, 
            topo_path,
            failure_id=self.failure_id
        )

        pairs_path = os.path.join(
            dataset_dir, "Pairs", pairs_name
        )
        self.pairs = _read_pairs_from_pickle(
            pairs_path
        )

        tm_path = os.path.join(
            dataset_dir, "TMs", tm_name
        )
        self.tm = _read_tms(
            tm_path,
            num_paths_per_pair
        )

        assert (self.tm.shape[0] == num_paths_per_pair * len(self.pairs)), \
            f"Assertion: repeated tms are expected to have \
                {num_paths_per_pair * len(self.pairs)}, but have {self.tm.shape[0]} only..."
        
        self.num_demands = len(self.pairs)
        self.node_ids_map = {node: i for (i, node) in enumerate(self.graph.nodes())}
        self.node_features = self.get_node_features()
    
    def get_edge_indices(self) -> torch.Tensor:
        src_lst = []
        dst_lst = []
        for (u, v) in self.graph.edges():
            x = self.node_ids_map[u]
            y = self.node_ids_map[v]
            src_lst.append(x)
            dst_lst.append(y)
        
        edge_indices = torch.tensor([src_lst, dst_lst], dtype=torch.int64)
        return edge_indices
    
    def get_node_features(self) -> torch.Tensor:
        # In degree for each node
        in_degrees = dict(self.graph.in_degree())
        degrees = torch.tensor(
            list(in_degrees.values()), 
            dtype=torch.float32
        ).reshape(-1, 1)
        edge_indices = self.get_edge_indices()

        link_caps = self.link_caps.clone()
        mask = torch.where(link_caps == 0)
        link_caps[mask] += 1e-20
    
        # Out link capacities for each node
        cap_sum_lst = []
        for node in self.graph.nodes():
            nid = self.node_ids_map[node]
            indices = (edge_indices[0] == nid).nonzero()
            cap_sum = torch.sum(link_caps[indices])
            cap_sum_lst.append(cap_sum)
        
        cap_sum_tensor = torch.tensor(
            cap_sum_lst, 
            dtype=torch.float32
        ).reshape(-1, 1)

        node_features = torch.cat(
            [degrees, cap_sum_tensor],
            dim=1
        )
        return node_features


def _read_graph_from_json(
    topology: str,
    topology_path: str,
    failure_id: Optional[int]
):
    with open(topology_path, "r") as f:
        json_data = json.load(f)
    
    graph = nx.readwrite.json_graph.node_link_graph(json_data)

    if failure_id is not None:
        #print(f"failure_id = {failure_id}")
        G_u = graph.to_undirected()
        G_u_edges = list(sorted(G_u.edges()))
        #for (i, e) in enumerate(G_u_edges):
        #    if G_u.edges[e]['capacity'] == 2.4:
        #        print(f"index = {i}")

        (u, v) = G_u_edges[failure_id]
        graph.edges[(u, v)]['capacity'] = 0
        graph.edges[(v, u)]['capacity'] = 0

    out_caps = defaultdict(float)
    for (u, v, data) in graph.edges(data=True):
        out_caps[u] += float(data['capacity'])

    link_caps = [
        float(data['capacity'])
        for (u, v, data) in graph.edges(data=True)
    ]

    #for (u, v, data) in graph.edges(data=True):
    #    if float(data['capacity']) < 1e-4:
    #        print(f"??? {u} {v}: {data['capacity']}")
    link_caps = torch.tensor(link_caps, dtype=torch.float32)

    #normed_link_caps = [
    #    float(data['capacity']) / out_caps[u]
    #    for (u, v, data) in graph.edges(data=True)
    #]
    #normed_link_caps = torch.tensor(normed_link_caps, dtype=torch.float32)
    #normed_link_caps = link_caps

    return graph, link_caps, link_caps


def _read_pairs_from_pickle(
    pairs_path: str
):
    with open(pairs_path, "rb") as f:
        pairs = pickle.load(f)
    return pairs


def _read_tms(
    tm_path: str,
    num_paths_per_pair: int = 4
):
    with open(tm_path, "rb") as f:
        tm = pickle.load(f)
        tm = tm.reshape(-1, 1).astype(np.float32)
        # FIXME: num_paths_per_pair
        tm = np.repeat(tm, repeats=num_paths_per_pair, axis=0)
    return torch.tensor(tm, dtype=torch.float32)
