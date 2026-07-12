#! /usr/bin/env python

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.dataset_loader.dataset_cluster import SingleClusterDataset
from lib.teal_actor import TealActor
from lib.teal_env import TealEnv


def _get_num_samples(data_dir):
    catalog_path = os.path.join(data_dir, "Catalog/0/catalog_file.txt")
    filenames = np.loadtxt(catalog_path, dtype="U", delimiter=",").reshape(-1, 3)
    return filenames.shape[0]


def _make_test_dataset(args):
    num_samples = _get_num_samples(args.data_dir)
    start = int(args.train_test_split * num_samples)
    end = num_samples
    if args.max_samples > 0:
        end = min(end, start + args.max_samples)
    if end <= start:
        raise ValueError(
            f"empty TEAL profiling range for {args.topo_name}: start={start}, end={end}"
        )

    dataset = SingleClusterDataset(
        args.data_dir,
        args.topo_name,
        cluster_id=0,
        num_paths_per_pair=args.num_paths_per_pair,
        start=start,
        end=end,
        use_opt=False,
    )
    return dataset, start, end, num_samples


def _make_teal_components(args, device):
    train_size = [0, 0]
    val_size = [0, 0]
    test_size = [0, 0]
    teal_env = TealEnv(
        obj=args.obj,
        topo=args.topo_name,
        problems=None,
        num_path=args.num_paths_per_pair,
        edge_disjoint=False,
        dist_metric="min-hop",
        rho=args.rho,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_failure=0,
        device=device,
    )
    teal_actor = TealActor(
        teal_env=teal_env,
        num_layer=args.layers,
        model_dir=args.model_dir,
        model_save=False,
        device=device,
        model_load=args.model_load,
    )
    teal_actor.eval()
    teal_env.training = False
    return teal_env, teal_actor


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summarize(latencies_s):
    if not latencies_s:
        raise ValueError("no TEAL latency samples collected")
    mean_s = float(statistics.mean(latencies_s))
    return {
        "num_timed_samples": len(latencies_s),
        "total_s": float(sum(latencies_s)),
        "mean_s": mean_s,
        "median_s": float(statistics.median(latencies_s)),
        "min_s": float(min(latencies_s)),
        "max_s": float(max(latencies_s)),
        "p90_s": _percentile(latencies_s, 90),
        "p95_s": _percentile(latencies_s, 95),
        "p99_s": _percentile(latencies_s, 99),
        "std_s": float(statistics.pstdev(latencies_s)),
        "samples_per_second": float(1.0 / mean_s) if mean_s > 0 else 0.0,
    }


def _append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def profile(args):
    if args.bsz != 1:
        raise ValueError("this TEAL profiler is intended for batch_size=1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset, start, end, total_samples = _make_test_dataset(args)
    loader = DataLoader(dataset, batch_size=args.bsz, shuffle=False)
    teal_env, teal_actor = _make_teal_components(args, device)

    p2e_matrix = dataset.pte
    teal_env.set_topo_info(p2e_matrix)
    teal_actor.reset_num_path_node(p2e_matrix.size(0))

    latencies_s = []
    seen = 0
    with torch.no_grad():
        for _, link_caps, tms, _ in loader:
            seen += 1
            link_cap = link_caps[0].to(device)
            tm = tms.squeeze(dim=-1)[0].to(device)

            _sync(device)
            start_time = time.perf_counter()
            teal_env.set_obs(link_cap, tm)
            obs = teal_env.get_obs()
            raw_action = teal_actor.act(obs)
            teal_env.step(raw_action, num_admm_step=args.admm_steps)
            _sync(device)

            if seen > args.warmup_samples:
                latencies_s.append(time.perf_counter() - start_time)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = _summarize(latencies_s)
    summary.update(
        {
            "method": "teal",
            "topo_name": args.topo_name,
            "data_dir": args.data_dir,
            "device": str(device),
            "batch_size": args.bsz,
            "warmup_samples": args.warmup_samples,
            "sample_start": start,
            "sample_end": end,
            "num_profile_samples": end - start,
            "num_total_samples": total_samples,
            "num_paths": int(dataset.pte.size(0)),
            "num_edges": int(dataset.pte.size(1)),
            "num_paths_per_pair": args.num_paths_per_pair,
            "layers": args.layers,
            "rho": args.rho,
            "admm_steps": args.admm_steps,
            "obj": args.obj,
            "model_load": args.model_load,
            "model_dir": os.path.realpath(args.model_dir),
            "num_model_parameters": sum(p.numel() for p in teal_actor.parameters()),
            "timestamp": timestamp,
        }
    )

    print(json.dumps(summary, indent=2))
    print(
        "FINAL_AVG_INFERENCE_TIME_S "
        f"topology={args.topo_name} method=teal "
        f"skipped={args.warmup_samples} "
        f"timed_samples={summary['num_timed_samples']} "
        f"avg_s={summary['mean_s']:.9f}"
    )

    csv_path = os.path.join(args.result_dir, "inference_profile_summary.csv")
    json_path = os.path.join(
        args.result_dir,
        f"{args.topo_name}_bs{args.bsz}_{timestamp}.json",
    )
    _append_csv(csv_path, summary)
    _write_json(json_path, {"results": [summary]})
    print(f"Wrote CSV summary to {csv_path}")
    print(f"Wrote JSON summary to {json_path}")


def make_args():
    parser = argparse.ArgumentParser(
        description="Profile TEAL batch-size-1 inference overhead on static test data."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--topo_name", required=True)
    parser.add_argument(
        "--result_dir",
        default=os.path.join(ROOT_DIR, "results", "profile", "teal"),
    )
    parser.add_argument("--num_paths_per_pair", type=int, default=4)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--train_test_split", type=float, default=0.75)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--warmup_samples", type=int, default=20)
    parser.add_argument("--admm-steps", dest="admm_steps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--obj", type=str, default="total_flow")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(ROOT_DIR, "teal-models"),
    )
    parser.add_argument(
        "--model-load",
        action="store_true",
        help="require and load weights from an objective-specific checkpoint",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    profile(make_args())
