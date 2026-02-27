import glob
import os
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loader.snapshot import SnapshotReader
from dataset_loader.cluster import ClusterLoader

from typing import Optional

class StaticDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        topo_name: str,
        num_paths_per_pair: int,
        use_opt: bool = True,
        failure_id: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.topo_name = topo_name
        self.num_paths_per_pair = num_paths_per_pair
        self.use_opt = use_opt
        self.failure_id = failure_id

        if topo_name == "meta_tor_a":
            # FIXME: do not hard coded
            self.tm_dir = "/workspace/NetAI/data/meta_tor_a_raw/TMs"
        else:
            self.tm_dir = os.path.join(
                dataset_dir, "TMs"
            )
        # Step 1: loading topology-related info
        loader = self.mk_cluster_loader()
        reader = loader.reader

        self.graph = reader.graph

        self.node_features = reader.node_features
        self.link_caps = reader.link_caps

        self.edge_index = reader.get_edge_indices()
        self.pij = loader.get_ksp_paths(
            self.num_paths_per_pair, reader.pairs
        )
        self.pte = loader.get_paths_to_edges_coo_tensor(self.pij)
        self.padded_edge_ids_per_path = loader.get_padded_edge_ids_per_path(
            self.pij, loader.edges_map
        )
        self.num_pairs = loader.num_pairs

        # Step 2: loading traffic matrices and optimal values
        self.tm_lst = self.load_tms(self.graph)
        self.opt_lst = self.load_opts(len(self.tm_lst))
    
    def load_tms(self, G):
        tm_paths = sorted(glob.glob(f"{self.tm_dir}/*.hist"))
        num_nodes = len(G.nodes())
        mask = (~np.eye(num_nodes, dtype=bool)).flatten()

        tm_lst = []
        for tm_path in tqdm.tqdm(tm_paths):
            with open(tm_path, "rb") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    tm = np.array(
                        [
                            float(_) for _ in line.split(b" ") if _
                        ],
                        dtype=np.float32
                    )
                    tm = tm / 10_000
                    tm = tm[mask]
                    tm = tm.reshape(-1, 1).astype(np.float32)
                    tm = np.repeat(tm, repeats=self.num_paths_per_pair, axis=0)
                    tm_lst.append(tm)
        return tm_lst

    def load_opts(self, num_tms):
        if self.use_opt:
            opt_dir = os.path.join(
                self.dataset_dir, "Opt", 
                f"{self.num_paths_per_pair}sp", 
                "0"
            )
            opt_name = (
                "opt_values.txt" 
                if self.failure_id is None
                else f"opt_values_failure_id{self.failure_id}.txt"
            )
            opt_path: str = os.path.join(opt_dir, opt_name)
            with open(opt_path, "r") as f:
                opts = np.loadtxt(f, dtype=np.float32).ravel()
            # TODO: truncation with `start` and `end`
        else:
            opts = np.zeros(num_tms, dtype=np.float32)
        return opts

    def mk_cluster_loader(self):
        manifest_path: str = os.path.join(
            self.dataset_dir, "Catalog/0", "catalog_file.txt"
        )
        file_names = np.loadtxt(
            manifest_path, 
            dtype="U",
            delimiter=","
        ).reshape(-1, 3)
        (topo_name, pairs_name, tm_name) = file_names[0]
        reader = SnapshotReader(
            self.dataset_dir,
            self.topo_name,
            topo_name, pairs_name, tm_name,
            self.num_paths_per_pair,
            failure_id=self.failure_id
        )
        loader = ClusterLoader(
            self.topo_name,
            reader,
            self.dataset_dir,
            cluster_id=0,
            num_paths_per_pair=self.num_paths_per_pair,
            use_recompute=False
        )
        return loader
    
    def __len__(self):
        return len(self.tm_lst)
    
    def __getitem__(self, idx):
        return (
            self.node_features,
                self.link_caps,
                    1,
                        self.tm_lst[idx],
                            self.opt_lst[idx]
        )