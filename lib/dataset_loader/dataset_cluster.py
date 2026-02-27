import glob
import os
import sys
import math
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch

from torch.utils.data import Dataset
from .snapshot import SnapshotReader
from .cluster import ClusterLoader

import tqdm

from typing import Optional

class SingleClusterDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        topo_name: str,
        cluster_id: int,
        num_paths_per_pair: int,
        start: int = 0,
        end: Optional[int] = None,
        use_opt: bool = True,
        failure_id: Optional[int] = None
    ):
        self.dataset_dir = dataset_dir
        self.topo_name = topo_name
        self.cluster_id = cluster_id
        self.num_paths_per_pair = num_paths_per_pair
        self.start = start
        self.end = end
        self.use_opt = use_opt
        self.failure_id = failure_id

        self.tm_lst = []
        self.node_features_lst = []
        self.opt_lst = []
        self.link_caps_lst = []
        
        manifest_path: str = os.path.join(
            dataset_dir, "Catalog", 
            f"{self.cluster_id}", "catalog_file.txt"
        )
        filenames = np.loadtxt(
            manifest_path, 
            dtype="U",
            delimiter=","
        ).reshape(-1, 3)

        filenames = filenames[start:end]
            
        self.opt_lst = self.load_opts(len(filenames))
        #if self.failure_id is not None and self.use_opt:
        #    opt_dir = os.path.join(
        #        self.dataset_dir, "Opt", 
        #        f"{self.num_paths_per_pair}sp", 
        #        f"{self.cluster_id}"
        #    )
        #    failure_opt_name = f"opt_values_failure_id{self.failure_id}_testing.txt" 
        #    failure_path = os.path.join(
        #        opt_dir, 
        #        failure_opt_name
        #    )
        #    with open(failure_path, "r") as f:
        #        failure_opts = np.loadtxt(f, dtype=np.float32).ravel()
        #    self.opt_lst = failure_opts
        self.load_sample_tuples(filenames, self.opt_lst)
    
    def load_opts(self, num_tms):
        # FIXME: for failure model
        if self.use_opt:
            opt_dir = os.path.join(
                self.dataset_dir, "Opt", 
                f"{self.num_paths_per_pair}sp", 
                f"{self.cluster_id}"
            )
            opt_name = (
                "opt_values.txt" 
                if self.failure_id is None
                else f"opt_values_failure_id{self.failure_id}.txt"
            )

            # A quick fix, by testing on the test dataset only.
            opt_name = "opt_values.txt"
            opt_path: str = os.path.join(opt_dir, opt_name)

            #print(f"opt_path = {opt_path}")
            with open(opt_path, "r") as f:
                opts = np.loadtxt(f, dtype=np.float32).ravel()

            opts = opts[self.start:self.end]
        else:
            opts = np.zeros(num_tms, dtype=np.float32)
        return opts
    
    def load_sample_tuples(self, filenames, opts):
        # TODO: `pred_tm_lst`
        # FIXME: use faster reader for large Meta topology.
        #print(opts)
        #print(f"len of filenames = {len(filenames)}, len opts = {len(opts)}")
        for (snapshot_tuple, opt_val) in tqdm.tqdm(zip(filenames, opts),
                                                   total=len(filenames)):
            (
                topo_filename, pairs_filename, tm_filename
            ) = snapshot_tuple
            reader = SnapshotReader(
                self.dataset_dir,
                self.topo_name,
                topo_filename,
                pairs_filename,
                tm_filename,
                self.num_paths_per_pair,
                failure_id=self.failure_id
            )
            self.tm_lst.append(reader.tm)
            self.node_features_lst.append(reader.node_features)
            self.link_caps_lst.append(reader.link_caps)

        cluster_loader = ClusterLoader(
            self.topo_name, 
            reader, 
            self.dataset_dir,
            self.cluster_id,
            self.num_paths_per_pair
        )
        self.graph = reader.graph
        self.edge_index = reader.get_edge_indices()
        self.pij = cluster_loader.get_ksp_paths(
            self.num_paths_per_pair,
            cluster_loader.reader.pairs
        )
        self.pte = cluster_loader.get_paths_to_edges_coo_tensor(self.pij)
        #self.ftp = cluster_loader.get_flow_to_paths_coo_tensor(self.pij)
        self.ftp = None
        self.fte = None

        self.padded_edge_ids_per_path = cluster_loader.get_padded_edge_ids_per_path(
            self.pij, 
            cluster_loader.edges_map
        )
        self.num_pairs = cluster_loader.num_pairs
        (n_paths, n_edges) = self.pte.size()
        self.lpnorm_lst = [
            1 for _ in range(len(self.tm_lst))
        ]

    def __len__(self):
        return len(self.tm_lst)
    
    def __getitem__(self, index):
        # TODO: using `pred_tm_lst`?
        return (
            self.node_features_lst[index],
                self.link_caps_lst[index],
                        self.tm_lst[index],  
                            self.opt_lst[index]
        )
