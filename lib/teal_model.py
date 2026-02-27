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
from torch.utils.data import DataLoader

from .teal_actor import TealActor
from .teal_env import TealEnv
from .utils import print_

from .dataset_loader.dataset_cluster import SingleClusterDataset

class Teal():
    def __init__(self, teal_env, teal_actor, lr, early_stop):
        """Initialize Teal model.

        Args:
            teal_env: teal environment
            num_layer: number of flowGNN layers
            lr: learning rate
            early_stop: whether to early stop
        """

        self.env = teal_env
        self.actor = teal_actor

        # TODO: tidy up these things
        
        DATA_DIR = "/workspace/NetAI/data_kaete/geant"
        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []

        self.train_loaders = [
            torch.utils.data.DataLoader(
                SingleClusterDataset(
                    DATA_DIR,
                    "geant",
                    0,
                    4,
                    start=0,
                    end=6000,
                ),
                batch_size=32,
                shuffle=False
            )
        ]
        self.val_loaders = [
            torch.utils.data.DataLoader(
                SingleClusterDataset(
                    DATA_DIR,
                    "geant",
                    0, 4, 
                    start=7000,
                    end=9000
                ),
                batch_size=32,
                shuffle=False
            )
        ]

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
            for train_loader in self.train_loaders:
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
                    reward, info = self.env.step(raw_action, num_admm_step=3)
                    rewards.append(reward.item() / opts[idx].item())
                pbar.set_postfix({'rel_loss_mean': '%.7f' % (np.mean(rewards)),
                                  'rel_loss_min': '%.7f' % (np.min(rewards)),
                                  '1th': '%.7f' % (np.percentile(rewards, 1))})
        return rewards
        #rewards = 0
        #for idx in range(self.env.idx_start, self.env.idx_stop):

        #    # get observation
        #    problem_dict = self.env.render()
        #    obs = self.env.get_obs()
        #    # get action
        #    raw_action = self.actor.act(obs)
        #    # get reward
        #    reward, info = self.env.step(raw_action)
        #    # show satisfied demand instead of total flow
        #    rewards += reward.item()/problem_dict['total_demand']\
        #        if self.env.obj == 'total_flow' else reward.item()
        #self.val_reward.append(
        #    rewards/(self.env.idx_stop - self.env.idx_start))

    def test(self, num_admm_step, output_header, output_csv, output_dir):
        """Test Teal model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.training = False
        self.env.reset('test')

        with open(output_csv, "a") as results:
            print_(",".join(output_header), file=results)

            runtime_list, obj_list = [], []
            loop_obj = tqdm(
                range(self.env.idx_start, self.env.idx_stop),
                desc="Testing: ")

            for idx in loop_obj:

                # get observation
                problem_dict = self.env.render()
                obs = self.env.get_obs()
                # get action
                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time
                # get reward
                reward, info = self.env.step(
                    raw_action, num_admm_step=num_admm_step)
                # add runtime in transforming, ADMM, rounding
                runtime += info['runtime']
                runtime_list.append(runtime)
                # show satisfied demand instead of total flow
                obj_list.append(
                    reward.item()/problem_dict['total_demand']
                    if self.env.obj == 'total_flow' else reward.item())

                # display avg runtime, obj
                loop_obj.set_postfix({
                    'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                    'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                    })

                # save solution matrix
                sol_mat = info['sol_mat']
                torch.save(sol_mat, os.path.join(
                    output_dir,
                    "{}-{}-{}-teal_objective-{}_{}-paths_"
                    "edge-disjoint-{}_dist-metric-{}_sol-mat.pt".format(
                        problem_dict['problem_name'],
                        problem_dict['traffic_model'],
                        problem_dict['traffic_seed'],
                        problem_dict['obj'],
                        problem_dict['num_path'],
                        problem_dict['edge_disjoint'],
                        problem_dict['dist_metric'])))

                PLACEHOLDER = ",".join("{}" for _ in output_header)
                result_line = PLACEHOLDER.format(
                    problem_dict['problem_name'],
                    problem_dict['num_node'],
                    problem_dict['num_edge'],
                    problem_dict['traffic_seed'],
                    problem_dict['scale_factor'],
                    problem_dict['traffic_model'],
                    problem_dict['total_demand'],
                    "Teal",
                    problem_dict['num_path'],
                    problem_dict['edge_disjoint'],
                    problem_dict['dist_metric'],
                    problem_dict['obj'],
                    reward,
                    runtime)
                print_(result_line, file=results)
                # break
