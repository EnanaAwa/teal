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
                    reward, info = self.env.step(raw_action)
                    rewards.append(reward.item() / opts[idx].item())
                pbar.set_postfix({'rel_loss_mean': '%.7f' % (np.mean(rewards)),
                                  'rel_loss_min': '%.7f' % (np.min(rewards)),
                                  '1th': '%.7f' % (np.percentile(rewards, 1))})
        return rewards

    def test(self, num_admm_step, output_dir):
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