#!/usr/bin/env python

import os
import sys
import datetime
import random

import numpy as np
import torch

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from teal_helper import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS

from lib.teal_env import TealEnv
from lib.teal_actor import TealActor
from lib.teal_model import Teal


HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "scale_factor",
    "tm_model",
    "total_demand",
    "algo",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "objective",
    "obj_val",
    "runtime",
]

OUTPUT_CSV_TEMPLATE = "teal-{}-{}.csv"


def benchmark(problems, output_csv, args):

    num_path, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS

    num_path = args.num_paths_per_pair

    obj, topo = args.obj, args.topo_name
    model_save = args.model_save
    device = torch.device(
        f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")

    # ========== load hyperparameters
    # env hyper-parameters
    train_size = [args.slice_train_start, args.slice_train_stop]
    val_size = [args.slice_val_start, args.slice_val_stop]
    test_size = [args.slice_test_start, args.slice_test_stop]
    # actor hyper-parameters
    num_layer = args.layers
    rho = args.rho
    # training hyper-parameters
    lr = args.lr
    early_stop = args.early_stop
    num_epoch = args.epochs
    batch_size = args.bsz
    num_sample = args.samples
    num_admm_step = args.admm_steps
    # testing hyper-parameters
    num_failure = args.failures

    # ========== init teal env, actor, model
    teal_env = TealEnv(
        obj=obj,
        topo=topo,
        problems=problems,
        num_path=num_path,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        rho=rho,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_failure=num_failure,
        device=device
    )
    teal_actor = TealActor(
        teal_env=teal_env,
        num_layer=num_layer,
        model_dir=args.model_dir,
        model_save=model_save,
        device=device,
        model_load=args.model_load,
    )
    teal = Teal(
        data_dir=args.data_dir,
        topo_name=args.topo_name,
        num_path=num_path,
        batch_size=batch_size,
        teal_env=teal_env,
        teal_actor=teal_actor,
        lr=lr,
        early_stop=early_stop,
        num_clusters=args.num_clusters,
        num_train_clusters=args.num_train_clusters,
        num_val_clusters=args.num_val_clusters,
        train_test_split=args.train_test_split,
        max_dataset_samples=args.max_dataset_samples,
        seed=args.seed,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = os.path.join(
        args.topo_name,
        "teal",
        f"{obj}_epoch{num_epoch}_b{batch_size}_lr{lr}_admm{num_admm_step}_seed{args.seed}-{timestamp}.json"
    )
    output_path = os.path.join(
        args.result_dir,
        exp_name
    )
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Teal log path: {output_path}")

    # ========== profile or train/test
    if args.profile_inference:
        teal.profile_inference(
            num_admm_step=num_admm_step,
            output_dir=args.profile_result_dir,
            warmup_samples=args.profile_warmup_samples
        )
    else:
        teal.train(
            num_epoch=num_epoch,
            batch_size=batch_size,
            num_sample=num_sample
        )
        teal.test(
            num_admm_step=num_admm_step,
            output_path=output_path,
            settings={
                "topo_name": args.topo_name,
                "objective": obj,
                "dataset_dir": os.path.realpath(args.data_dir),
                "result_dir": os.path.realpath(args.result_dir),
                "model_dir": os.path.realpath(args.model_dir),
                "checkpoint_path": os.path.realpath(teal_actor.model_fname),
                "model_load": args.model_load,
                "model_save": model_save,
                "device": str(device),
                "seed": args.seed,
                "epochs": num_epoch,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_paths_per_pair": num_path,
                "num_clusters": args.num_clusters,
                "num_train_clusters": args.num_train_clusters,
                "num_val_clusters": args.num_val_clusters,
                "train_test_split": args.train_test_split,
                "max_dataset_samples": args.max_dataset_samples,
                "layers": num_layer,
                "rho": rho,
                "samples": num_sample,
                "admm_steps": num_admm_step,
            },
        )
    return


if __name__ == '__main__':
    args, output_csv, problems = get_args_and_problems(OUTPUT_CSV_TEMPLATE)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, output_csv, args)
