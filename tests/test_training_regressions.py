import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from lib.teal_actor import TealActor
from lib.teal_model import Teal, _load_static_dataset


class _Progress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values):
        pass


def _progress(iterable, **kwargs):
    return _Progress(iterable)


class _TrainLoader:
    def __init__(self, num_batches):
        self.dataset = SimpleNamespace(pte=torch.zeros(1, 1))
        self.batches = [
            (
                None,
                torch.ones(1, 1),
                torch.ones(1, 1, 1),
                torch.ones(1),
            )
            for _ in range(num_batches)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Actor:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.train_calls = 0
        self.eval_calls = 0
        self.save_calls = 0

    def train(self):
        self.train_calls += 1

    def eval(self):
        self.eval_calls += 1

    def reset_num_path_node(self, num_path_nodes):
        pass

    def evaluate(self, obs):
        return torch.zeros(1), self.parameter.reshape(1)

    def save_model(self):
        self.save_calls += 1


class _Environment:
    def __init__(self):
        self.training = False
        self.training_during_step = []
        self.reset_modes = []

    def reset(self, mode):
        self.reset_modes.append(mode)

    def set_topo_info(self, p2e_matrix):
        pass

    def set_obs(self, link_cap, tm):
        pass

    def get_obs(self):
        return torch.ones(1)

    def step(self, raw_action, num_sample):
        self.training_during_step.append(self.training)
        return torch.ones(1), {}


class _Optimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        self.parameter.grad = None

    def step(self):
        self.step_calls += 1


class _StaticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        topo_name,
        cluster_id,
        num_paths_per_pair,
        start,
        end,
    ):
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        return index


class TrainingRegressionTest(unittest.TestCase):
    def test_updates_once_per_minibatch_and_restores_training_each_epoch(self):
        actor = _Actor()
        environment = _Environment()
        optimizer = _Optimizer(actor.parameter)

        teal = Teal.__new__(Teal)
        teal.actor = actor
        teal.env = environment
        teal.actor_optimizer = optimizer
        teal.train_loaders = [_TrainLoader(num_batches=2)]

        validation_calls = []

        def validate():
            validation_calls.append(True)
            actor.eval()
            environment.training = False

        teal.val = validate

        with mock.patch("lib.teal_model.tqdm", new=_progress):
            teal.train(num_epoch=2, batch_size=1, num_sample=1)

        self.assertEqual(optimizer.zero_grad_calls, 4)
        self.assertEqual(optimizer.step_calls, 4)
        self.assertEqual(actor.train_calls, 2)
        self.assertEqual(actor.save_calls, 1)
        self.assertEqual(len(validation_calls), 2)
        self.assertEqual(environment.reset_modes, ["train", "train"])
        self.assertEqual(environment.training_during_step, [True] * 4)

    def test_static_split_is_seeded(self):
        def split(seed):
            with mock.patch("lib.teal_model._get_num_samples", return_value=20), mock.patch(
                "lib.teal_model.SingleClusterDataset", new=_StaticDataset
            ):
                train, validation, test = _load_static_dataset(
                    "unused",
                    "geant",
                    num_paths=4,
                    batch_size=4,
                    seed=seed,
                )
            return (
                list(train[0].sampler.indices),
                list(validation[0].sampler.indices),
                test[0].dataset.start,
                test[0].dataset.end,
            )

        split_42 = split(42)
        self.assertEqual(split_42, split(42))
        self.assertNotEqual(split_42[:2], split(7)[:2])
        self.assertEqual(split_42[2:], (15, 20))


class CheckpointRegressionTest(unittest.TestCase):
    def make_actor(self, objective):
        actor = TealActor.__new__(TealActor)
        torch.nn.Module.__init__(actor)
        actor.env = SimpleNamespace(obj=objective)
        return actor

    def test_checkpoint_name_includes_objective(self):
        mlu_path = self.make_actor("mlu").model_full_fname(
            "/models", "geant", num_layer=6, std=1
        )
        flow_path = self.make_actor("total_flow").model_full_fname(
            "/models", "geant", num_layer=6, std=1
        )
        self.assertNotEqual(mlu_path, flow_path)
        self.assertEqual(
            os.path.basename(mlu_path),
            "geant_mlu_flowGNN-6_std-False.pt",
        )

    def test_explicit_missing_checkpoint_is_an_error(self):
        actor = self.make_actor("mlu")
        with tempfile.TemporaryDirectory() as directory:
            actor.model_fname = os.path.join(directory, "missing.pt")
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                actor.load_model()

    def test_save_creates_checkpoint_directory(self):
        actor = self.make_actor("mlu")
        actor.model_save = True
        with tempfile.TemporaryDirectory() as directory:
            actor.model_fname = os.path.join(directory, "nested", "model.pt")
            actor.save_model()
            self.assertTrue(os.path.exists(actor.model_fname))


if __name__ == "__main__":
    unittest.main()
