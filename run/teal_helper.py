from collections import defaultdict
from glob import iglob

import argparse
import os
import sys

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.config import TL_DIR, TOPOLOGIES_DIR, TM_DIR

PROBLEM_NAMES = [
    'B4.json',
    'UsCarrier.json',
    'Kdl.json',
    'ASN2k.json',
]
TM_MODELS = [
    "real",
    "toy",
]
SCALE_FACTORS = [1.0]
OBJ_STRS = ["total_flow", "min_max_link_util", "mlu"]

PATH_FORM_HYPERPARAMS = (4, False, "min-hop")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES
    for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join(TOPOLOGIES_DIR, problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob(
            "{}/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "{}/holdout/{}/{}*_traffic-matrix.pkl".format(
                TM_DIR, model, problem_name
            )
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)]\
                .append(
                    (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))


def get_problems(args):
    #FIXME: bypass all fucking problems for now
    #if (args.topo, args.tm_model, args.scale_factor) not in GROUPED_BY_PROBLEMS:
    #    raise Exception('Traffic matrices not found')
    #problems = []
    #for topo_fname, tm_fname in GROUPED_BY_PROBLEMS[
    #        (args.topo, args.tm_model, args.scale_factor)]:
    #    problems.append((args.topo, topo_fname, tm_fname))
    #return problems
    return None


def get_args_and_problems(formatted_fname_template, additional_args=[]):
    parser = argparse.ArgumentParser()

    # Problems arguments
    parser.add_argument(
        "--dry-run", dest="dry_run", default=False, action="store_true",
        help="list problems to run")
    parser.add_argument(
        "--obj", type=str, default='total_flow', choices=OBJ_STRS,
        help="objective function")
    parser.add_argument(
        "--tm-model", type=str, default='real', choices=TM_MODELS,
        help="traffic matrix model")
    #parser.add_argument(
    #    "--topo", type=str, required=True, choices=PROBLEM_NAMES,
    #    help="network topology")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="dataset directory containing Catalog, TMs, Opt, and topology data",
    )
    parser.add_argument(
        "--topo_name", 
        type=str, 
        required=True, 
        #choices=PROBLEM_NAMES,
        help="network topology"
    )
    parser.add_argument(
        "--num_paths_per_pair", 
        type=int, 
        default=4,
    )
    parser.add_argument('--num-clusters', type=int, default=50)
    parser.add_argument('--num-train-clusters', type=int, default=30)
    parser.add_argument('--num-val-clusters', type=int, default=5)
    parser.add_argument('--train-test-split', type=float, default=0.75)
    parser.add_argument(
        '--max-dataset-samples', type=int, default=0,
        help='limit static data or samples per dynamic cluster for smoke tests; 0 uses all data')
    parser.add_argument(
        "--scale-factor", type=float, default=1.0, choices=SCALE_FACTORS,
        help="traffic matrix scale factor")
    parser.add_argument(
        '--devid', type=int, default=0,
        help='GPU device id')
    parser.add_argument(
        '--model-save', nargs='?', const=True, default=False, type=parse_bool,
        help='whether to save model')
    parser.add_argument(
        '--model-load', nargs='?', const=True, default=False, type=parse_bool,
        help='load weights from an existing checkpoint for the same objective')
    parser.add_argument(
        '--model-dir', type=str, default=os.path.join(TL_DIR, 'teal-models'),
        help='directory for objective-specific model checkpoints')
    parser.add_argument(
        '--result-dir', type=str, default=os.path.join(TL_DIR, 'results'),
        help='root directory for JSON experiment results')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Python, NumPy, and PyTorch random seed')

    # env hyper-parameters
    parser.add_argument(
        '--slice-train-start', type=int, default=0,
        help="start index of training")
    parser.add_argument(
        '--slice-train-stop', type=int, default=20,
        help="end index of training")
    parser.add_argument(
        '--slice-val-start', type=int, default=20,
        help="start index of validation")
    parser.add_argument(
        '--slice-val-stop', type=int, default=28,
        help="end index of validation")
    parser.add_argument(
        '--slice-test-start', type=int, default=28,
        help="start index of testing")
    parser.add_argument(
        '--slice-test-stop', type=int, default=36,
        help="end index of testing")

    # actor hyper-parameters
    parser.add_argument(
        '--layers', type=int, default=6,
        help='number of flowGNN layers')
    parser.add_argument(
        '--rho', type=float, default=1.0,
        help='rho in ADMM')

    # training hyper-parameters
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='number of training epochs')
    parser.add_argument(
        '--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument(
        '--samples', type=int, default=5,
        help='number of COMA samples')
    parser.add_argument(
        '--admm-steps', type=int, default=3,
        help='number of ADMM steps')
    parser.add_argument(
        '--early-stop', nargs='?', const=True, default=False, type=parse_bool,
        help='whether to stop early')

    # testing hyper-parameters
    parser.add_argument(
        '--failures', type=int, default=0, help='number of edge failures')
    parser.add_argument(
        "--profile-inference",
        dest="profile_inference",
        default=False,
        action="store_true",
        help="Profile TEAL end-to-end inference time on the test loaders.")
    parser.add_argument(
        "--profile-warmup-samples",
        dest="profile_warmup_samples",
        type=int,
        default=20,
        help="Number of initial test samples to skip while profiling.")
    parser.add_argument(
        "--profile-result-dir",
        dest="profile_result_dir",
        type=str,
        default=os.path.join(TL_DIR, "results", "profile", "teal"),
        help="Directory for TEAL inference profile logs.")

    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()

    slice_str = "all"  # "slice_" + "_".join(str(i) for i in args.slices)
    formatted_fname_substr = formatted_fname_template.format(
        args.obj, slice_str)
    return args, formatted_fname_substr, get_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
