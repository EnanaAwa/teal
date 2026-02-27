
import os
import sys
import tqdm

import numpy as np
import networkx as nx
import pickle as pkl
import torch
from scipy.sparse import csr_matrix
from torch.nn.utils.rnn import pad_sequence

from .snapshot import SnapshotReader
from itertools import islice
from typing import (
    Dict,
    Optional,
    Tuple,
    List
)

CommodityType = Tuple[int, int]
PathType = List[int]
PathMapType = Dict[CommodityType, List[PathType]]

#FIXME: `ksp paths` and `paths_to_edges` may require recomputation ...
class ClusterLoader(object):
    def __init__(
        self,
        topology: str,
        reader: SnapshotReader,
        cluster_dir: str,
        cluster_id: int,
        num_paths_per_pair: int,
        use_recompute: bool = True
    ):
        self.topology = topology
        self.reader = reader
        self.cluster_id = cluster_id
        self.use_recompute = use_recompute

        self.edges_map = {
            (i, j): eid
            for (eid, (i, j)) in enumerate(self.reader.graph.edges())
        }

        # FIXME: refactor `NUM_PATHS_PER_PAIR` and `CLUSTER_DIR`
        self.num_paths_per_pair = num_paths_per_pair
        self.cluster_dir: str = cluster_dir
    
    def get_ksp_paths(
        self, 
        k,
        pairs
    ):
        filepath: str = os.path.join(
            self.cluster_dir, 
            "Paths", 
            f"{self.num_paths_per_pair}_paths_cluster_{self.cluster_id}.pkl"
        )
        try:
            with open(filepath, "rb") as f:
                pij = pkl.load(f)
                self.num_pairs = len(pij)
            return pij
        except:
            pij = dict()
            for (src, dst) in tqdm.tqdm(pairs):
                all_paths = _get_ksp_per_pair(
                    self.reader.graph,
                    src, dst, 
                    k,
                    weight=None
                )
                while len(all_paths) != k:
                    all_paths.append(all_paths[0])

                pij[(src,dst)] = [
                    _to_edge_tuple(path)
                    for (i, path) in enumerate(all_paths)
                ]
            self.num_pairs = len(pij)

            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "wb") as f:
                pkl.dump(pij, f)
            return pij
        
    def get_padded_edge_ids_per_path(
        self,
        pij: PathMapType,
        edges_map
    ) -> torch.Tensor:
        edge_ids_path: str = os.path.join(
            self.cluster_dir,
            "Paths",
            f"{self.num_paths_per_pair}_paths_cluster_{self.cluster_id}.pkl"
        )
        try:
            padded_edge_ids_per_path = torch.load(edge_ids_path)
        except:
            #print(f"Directly loading from {edge_ids_path} failed ..." 
            #       " Trying to figure out from path_dicts pij ...")
            paths_edges_lst = []
            for key in pij.keys():
                for path in pij[key]:
                    edges_lst = [
                        edges_map[e]
                        for e in path
                    ]
                    paths_edges_lst.append(
                        torch.tensor(
                            edges_lst, 
                            dtype=torch.int32
                        )
                    )
            padded_edge_ids_per_path = pad_sequence(
                paths_edges_lst,
                batch_first=True,
                padding_value=-1
            ).to(torch.int64)

            #torch.save(
            #    padded_edge_ids_per_path,
            #    edge_ids_path
            #)
        return padded_edge_ids_per_path
    
    def get_paths_to_edges_coo_tensor(
        self,
        pij: PathMapType
    ) -> torch.Tensor:
        pte_path: str = os.path.join(
            self.cluster_dir,
            "P2E",
            f"{self.num_paths_per_pair}_paths_cluster_{self.cluster_id}.pkl"
        )
        if os.path.exists(pte_path):
            (
                paths_to_edges
            ) = torch.load(pte_path)
            #paths_to_edges = torch.sparse_coo_tensor()
        else:
            if not os.path.exists(os.path.dirname(pte_path)):
                os.makedirs(os.path.dirname(pte_path))

            paths_arr = []
            path_to_commodity = dict()
            path_to_idx = dict()
            cntr = 0
            
            for key in pij.keys():
                idx = 0
                i, j = key
                for p in pij[(i, j)]:
                    p_ = [self.edges_map[e] for e in p]
                    p__ = np.zeros((len(self.reader.graph.edges()),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    path_to_commodity[cntr] = (i, j)
                    path_to_idx[cntr] = idx
                    cntr += 1
                    idx += 1
                    # commodities.append((i, j))
                
            paths_to_edges = np.stack(paths_arr)
            paths_to_edges = csr_matrix(paths_to_edges)
            paths_to_edges = paths_to_edges.tocoo()
            paths_to_edges = torch.sparse_coo_tensor(
                np.vstack((paths_to_edges.row, paths_to_edges.col)), 
                torch.FloatTensor(paths_to_edges.data), 
                torch.Size(paths_to_edges.shape)
            )
            torch.save(paths_to_edges, pte_path)
        return paths_to_edges
    
    def get_flow_to_paths_coo_tensor(
        self,
        pij: PathMapType
    ) -> torch.Tensor:
        row_idx = []
        col_idx = []

        num_paths = 0
        for (i, (_, paths)) in enumerate(pij.items()):
            k = len(paths)
            row_idx.extend([i for _ in range(len(paths))])
            col_idx.extend([j for j in range(num_paths, num_paths + k)])
            num_paths += k
        
        row_idx = torch.tensor(row_idx, dtype=torch.int64)
        col_idx = torch.tensor(col_idx, dtype=torch.int64)

        indices = torch.stack([row_idx, col_idx], dim=0)
        values = torch.ones_like(row_idx, dtype=torch.float32)

        flow_2_paths = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(self.num_pairs, num_paths)
        )
        return flow_2_paths


def _get_ksp_per_pair(
    graph, 
    src, dst,
    k,
    weight=None
):
    return list(
        islice(
            nx.shortest_simple_paths(
                graph,
                src, dst,
                weight=weight
            ), k
        )
    )

def _to_edge_tuple(path: List[int]):
    return list(zip(path, path[1:]))
