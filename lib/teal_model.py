import pickle
import time
import json
import sys
import os
import numpy as np
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler
)

from .teal_actor import TealActor
from .teal_env import TealEnv
from .utils import print_

from .dataset_loader.dataset_cluster import SingleClusterDataset

class Teal():
    def __init__(
        self, 
        data_dir,
        topo_name,
        num_path,
        batch_size,
        teal_env, 
        teal_actor, 
        lr, 
        early_stop):
        """Initialize Teal model.

        Args:
            teal_env: teal environment
            num_layer: number of flowGNN layers
            lr: learning rate
            early_stop: whether to early stop
        """

        self.data_dir = data_dir
        self.topo_name = topo_name
        self.num_path = num_path
        self.batch_size = batch_size
        self.env = teal_env
        self.actor = teal_actor


        # TODO: tidy up these hyperparameters
        NUM_CLUSTERS = 50
        NUM_TRAIN_CLUSTERS = 30

        if topo_name == "DynGEANT":
            (
                self.train_loaders, \
                    self.val_loaders, \
                        self.test_loaders
            ) = _load_dyn_dataset(
                self.data_dir,
                self.topo_name,
                self.num_path,
                NUM_CLUSTERS,
                NUM_TRAIN_CLUSTERS,
                self.batch_size
            )
        # TODO: load static dataset
        else:
            (
                self.train_loaders, \
                    self.val_loaders, \
                        self.test_loaders
            ) = _load_static_dataset(
                self.data_dir,
                self.topo_name,
                self.num_path,
                self.batch_size
            )


        # init optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # early stop when val result no longer changes
        self.early_stop = early_stop
        if self.early_stop:
            self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample):
        """Train Teal model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
        """
        self.env.training = True

        for epoch in range(num_epoch):
            for train_loader in tqdm(self.train_loaders):
                p2e_matrix = train_loader.dataset.pte
                self.env.set_topo_info(p2e_matrix)
                self.actor.reset_num_path_node(p2e_matrix.size(0))
                pbar = tqdm(train_loader, total=len(train_loader))

                loss_val_mean = []
                for (_, link_caps, tms, opts) in pbar:
                    batch_size = tms.size(0)
                    loss = 0
                    tms = tms.squeeze(dim=-1)
                    for idx in range(batch_size):
                        tm = tms[idx,:]
                        link_cap = link_caps[idx,:]

                        self.env.set_obs(link_cap, tm)

                        obs = self.env.get_obs()
                        raw_action, log_probs = self.actor.evaluate(obs)
                        reward, info = self.env.step(
                            raw_action,
                            num_sample=num_sample
                        )
                        loss += (-log_probs * reward).mean()
                        loss_val_mean.append(reward.mean().item() / opts[idx].item())
                
                    pbar.set_postfix({'loss': '%.5f' % (np.mean(loss_val_mean))})
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
            
            self.val()

            #self.env.reset('train')
            #ids = range(self.env.idx_start, self.env.idx_stop)
            #loop_obj = tqdm(
            #    [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
            #    desc=f"Training epoch {epoch}/{num_epoch}: ")

            #for idx in loop_obj:
            #    loss = 0
            #    for _ in idx:
            #        torch.cuda.empty_cache()

            #        # get observation
            #        obs = self.env.get_obs()
            #        # get action
            #        raw_action, log_probability = self.actor.evaluate(obs)
            #        # get reward
            #        reward, info = self.env.step(
            #            raw_action, num_sample=num_sample)
            #        loss += -(log_probability*reward).mean()

            #    self.actor_optimizer.zero_grad()
            #    loss.backward()
            #    self.actor_optimizer.step()
            #    # break

            # early stop
            #if self.early_stop:
            #    self.val()
            #    if len(self.val_reward) > 20 and abs(
            #            sum(self.val_reward[-20:-10])/10
            #            - sum(self.val_reward[-10:])/10) < 0.0001:
            #        break
        
        #FIXME:
        #self.actor.save_model()

    def val(self):
        """Validating Teal model."""

        self.actor.eval()
        self.env.training = False
        self.env.reset('val')

        rewards = 0

        rewards = []
        for val_loader in self.val_loaders:
            p2e_matrix = val_loader.dataset.pte
            self.env.set_topo_info(p2e_matrix)
            self.actor.reset_num_path_node(p2e_matrix.size(0))
            pbar = tqdm(val_loader, total=len(val_loader))

            for (_, link_caps, tms, opts) in pbar:
                batch_size = tms.size(0)
                tms = tms.squeeze(dim=-1)
                for idx in range(batch_size):
                    tm = tms[idx,:]
                    link_cap = link_caps[idx,:]

                    self.env.set_obs(link_cap, tm)

                    obs = self.env.get_obs()
                    raw_action = self.actor.act(obs)
                    reward, info = self.env.step(raw_action, num_admm_step=0)
                    rewards.append(reward.item() / opts[idx].item())
                pbar.set_postfix({'rel_loss_mean': '%.7f' % (np.mean(rewards)),
                                  'rel_loss_min': '%.7f' % (np.min(rewards)),
                                  '1th': '%.7f' % (np.percentile(rewards, 1))})
        return rewards

    def test(self, num_admm_step, output_path):
        """Test Teal model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.training = False
        #self.env.reset('test')

        reward_lst = []
        for test_loader in self.test_loaders:
            p2e_matrix = test_loader.dataset.pte
            self.env.set_topo_info(p2e_matrix)
            self.actor.reset_num_path_node(p2e_matrix.size(0))
            pbar = tqdm(test_loader, total=len(test_loader))
            for (_, link_caps, tms, opts) in pbar:
                batch_size = tms.size(0)
                tms = tms.squeeze(dim=-1)
                for idx in range(batch_size):
                    tm = tms[idx,:]
                    link_cap = link_caps[idx,:]

                    self.env.set_obs(link_cap, tm)

                    obs = self.env.get_obs()
                    raw_action = self.actor.act(obs)
                    reward, info = self.env.step(raw_action, num_admm_step=num_admm_step)
                    reward_lst.append(reward.item() / opts[idx].item())
                pbar.set_postfix({'rel_loss_mean': '%.7f' % (np.mean(reward_lst)),
                                  'rel_loss_min': '%.7f' % (np.min(reward_lst)),
                                  '1th': '%.7f' % (np.percentile(reward_lst, 1))})
        _get_percentiles(
            reward_lst,
            [1, 5, 10, 25, 50, 75, 90, 99]
        )
        print(f"Saving teal results to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(
                {"results": reward_lst},
                f
            )

    @torch.no_grad()
    def profile_inference(
        self,
        num_admm_step,
        output_dir="/workspace/NetAI/KaeTE/results/profile/teal",
        warmup_samples=20
    ):
        """Profile TEAL end-to-end inference time on test loaders.

        The timed region covers one sample's TEAL inference path:
        set_obs -> actor.act -> env.step. Topology setup and DataLoader
        iteration are intentionally outside the timing window.
        """
        self.actor.eval()
        self.env.training = False

        latencies_s = []
        samples_seen = 0

        for test_loader in self.test_loaders:
            p2e_matrix = test_loader.dataset.pte
            self.env.set_topo_info(p2e_matrix)
            self.actor.reset_num_path_node(p2e_matrix.size(0))
            self._sync_device()

            for (_, link_caps, tms, _) in test_loader:
                batch_size = tms.size(0)
                tms = tms.squeeze(dim=-1)
                for idx in range(batch_size):
                    tm = tms[idx, :]
                    link_cap = link_caps[idx, :]

                    self._sync_device()
                    start_time = time.perf_counter()
                    self.env.set_obs(link_cap, tm)
                    obs = self.env.get_obs()
                    raw_action = self.actor.act(obs)
                    self.env.step(raw_action, num_admm_step=num_admm_step)
                    self._sync_device()
                    elapsed_s = time.perf_counter() - start_time

                    samples_seen += 1
                    if samples_seen > warmup_samples:
                        latencies_s.append(elapsed_s)

        if not latencies_s:
            raise RuntimeError(
                f"No timed TEAL samples. warmup_samples={warmup_samples}, "
                f"samples_seen={samples_seen}"
            )

        profile_log = {
            "topo_name": self.topo_name,
            "method": "teal",
            "dataset_dir": self.data_dir,
            "device": str(self.env.device),
            "batch_size": self.batch_size,
            "num_test_loaders": len(self.test_loaders),
            "num_seen_samples": samples_seen,
            "num_timed_samples": len(latencies_s),
            "warmup_samples": warmup_samples,
            "measure": "teal_end_to_end",
            "timed_region": "set_obs -> actor.act -> env.step",
            "topology_setup_included": False,
            "admm_steps": num_admm_step,
            "mean_s": float(np.mean(latencies_s)),
            "median_s": float(np.median(latencies_s)),
            "min_s": float(np.min(latencies_s)),
            "max_s": float(np.max(latencies_s)),
            "p90_s": float(np.percentile(latencies_s, 90)),
            "p95_s": float(np.percentile(latencies_s, 95)),
            "p99_s": float(np.percentile(latencies_s, 99)),
            "std_s": float(np.std(latencies_s)),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }

        print(
            "FINAL_AVG_INFERENCE_TIME_S "
            f"topology={self.topo_name} method=teal "
            f"skipped={warmup_samples} "
            f"timed_samples={profile_log['num_timed_samples']} "
            f"avg_s={profile_log['mean_s']:.9f}"
        )
        self._save_profile_logs(profile_log, output_dir)
        return profile_log

    def _save_profile_logs(self, profile_log, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{self.topo_name}_bs{self.batch_size}_"
            f"{profile_log['timestamp']}_trainer_profile.json"
        )
        with open(output_path, "w") as f:
            json.dump({"results": [profile_log]}, f, indent=2)
        print(f"Saved TEAL inference profile to: {output_path}")

    def _sync_device(self):
        if self.env.device.type == "cuda":
            torch.cuda.synchronize(self.env.device)
        

def _get_percentiles(lst, p_lst):
    print(f"rel loss mean = {np.mean(lst)}, min = {np.min(lst)}")
    for p in p_lst:
        print(f"{p}-th: {np.percentile(lst, p)}")
    print(f"max = {np.max(lst)}")

def _load_dyn_dataset(
    data_dir,
    topo_name,
    num_paths,
    num_clusters,
    num_train_clusters,
    batch_size: int = 16,
    num_val_clusters: int = 5
):
    
    num_train = (
        num_train_clusters - \
            num_val_clusters
    )
    train_loaders = []
    for i in range(0, num_train):
        dataset = SingleClusterDataset(
            data_dir,
            topo_name,
            cluster_id=i,
            num_paths_per_pair=num_paths
        )
        train_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
        )
    
    val_loaders = []
    for i in range(num_train, num_train_clusters):
        dataset = SingleClusterDataset(
            data_dir,
            topo_name,
            cluster_id=i,
            num_paths_per_pair=num_paths
        )
        val_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
        )

    test_loaders = []
    for i in range(num_train_clusters, num_clusters):
        dataset = SingleClusterDataset(
            data_dir,
            topo_name,
            cluster_id=i,
            num_paths_per_pair=num_paths
        )
        test_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
        )
    
    return train_loaders, val_loaders, test_loaders


def _load_static_dataset(
    data_dir,
    topo_name,
    num_paths,
    batch_size,
    train_test_split: float = 0.75
):
    num_samples = _get_num_samples(data_dir)
    print(f"topo_name: {topo_name}, num_samples: {num_samples}")
    num_train = int(train_test_split * num_samples)

    train_dataset = SingleClusterDataset(
        data_dir,
        topo_name,
        0,
        num_paths_per_pair=num_paths,
        start=0,
        end=num_train
    )
    
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(num_train).tolist()

    num_train_samples = int(0.8 * num_train)
    train_indices = indices[:num_train_samples]
    eval_indices = indices[num_train_samples:]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices)
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(eval_indices)
    )

    test_dataset = SingleClusterDataset(
        data_dir,
        topo_name,
        0,
        num_paths_per_pair=num_paths,
        start=num_train,
        end=num_samples
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return [train_loader], [eval_loader], [test_loader]
    

def _get_num_samples(
    data_dir
):
    catalog_path = os.path.join(
        data_dir,
        "Catalog/0",
        "catalog_file.txt"
    )
    filenames = np.loadtxt(
        catalog_path, 
        dtype="U",
        delimiter=","
    ).reshape(-1, 3)
    return filenames.shape[0]
