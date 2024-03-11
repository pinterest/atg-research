from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import NamedTuple

import inspect
import os
import tempfile
import unittest
import uuid
from unittest import mock
from unittest.mock import call

import numpy as np
import pytest
import torch
from omnisearchsage.common.solver.basic_solver import BasicSolver
from omnisearchsage.common.utils.lr_scheduler import get_constant_schedule

if TYPE_CHECKING:
    from torch import Tensor
    from torch import nn


class DummyBatch(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor


class FakeDataLoader(object):
    def __init__(self, data):
        self.data = data
        # for repr
        self.dataset = "dataset"

    def __iter__(self):
        for d in self.data:
            yield DummyBatch(*[torch.tensor(v, dtype=torch.float32).reshape(1, 1) for v in d])

    def __len__(self):
        return len(self.data)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super(FakeModel, self).__init__()
        self.matrix = torch.nn.Linear(4, 2)
        self.count = 0

    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        self.count += 1
        feat = torch.cat(feats, 1).unsqueeze(0).to(self.matrix.weight.device)
        loss = self.matrix(feat).float().mean()
        return {BasicSolver.TOTAL_LOSS: loss}

    def __repr__(self) -> str:
        return "FakeModel@FooBar"

    def write_summary(self, summary_writer, epoch, iteration, total_iterations):
        pass


class FakeModel2(FakeModel):
    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        loss = super().forward(feats)[BasicSolver.TOTAL_LOSS]
        return {BasicSolver.TOTAL_LOSS: loss, BasicSolver.LOSS_COMPONENTS: {"loss_t": loss}}


class FakeModel3(FakeModel):
    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        loss = super().forward(feats)[BasicSolver.TOTAL_LOSS]
        return {BasicSolver.TOTAL_LOSS: loss, BasicSolver.LOSS_COMPONENTS: [loss]}


class FakeModel4(FakeModel):
    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        loss = super().forward(feats)[BasicSolver.TOTAL_LOSS]
        return {BasicSolver.TOTAL_LOSS: loss * float("nan"), BasicSolver.LOSS_COMPONENTS: [loss]}


class BasicSolverTest(unittest.TestCase):
    def test_run_eval_no_eval_fn(self) -> None:
        test_model = FakeModel()
        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        for eval_fn in [None, mock.MagicMock()]:
            solver = BasicSolver(
                model=test_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                batch_size=16,
                train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
                model_forward_func=lambda m, b: m(b),
                eval_func=eval_fn,
                eval_every_n_iter=20,
                iterations=45,
            )

            run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            solver.execute([0, 1], run_dir)

            if eval_fn is not None:
                eval_fn.assert_has_calls(
                    calls=[
                        mock.call(
                            run_dir=run_dir, eval_every_n_iter=20, summary_writer=solver.summary_writer, iteration=20
                        ),
                        mock.call(
                            run_dir=run_dir, eval_every_n_iter=20, summary_writer=solver.summary_writer, iteration=40
                        ),
                        mock.call(
                            run_dir=run_dir, eval_every_n_iter=20, summary_writer=solver.summary_writer, iteration=45
                        ),
                    ]
                )

    def test_snapshot(self) -> None:
        m = FakeModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        solver = BasicSolver(
            model=m,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=130,
            snapshot_every_n_iter=40,
            snapshot_filename_prefix="boby",
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)

        # test snapshot exists
        for snapshot_iter in [40, 80, 120]:
            first_snapshot = os.path.join(run_dir, "snapshots/boby_iter_{}.pth".format(snapshot_iter))
            self.assertTrue(os.path.exists(first_snapshot))

    def test_snapshot_savedir(self) -> None:
        m = FakeModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)
        snapshot_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver = BasicSolver(
            model=m,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=130,
            snapshot_every_n_iter=40,
            snapshot_filename_prefix="boby",
            snapshot_dir=snapshot_dir,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)

        # test snapshot exists
        for snapshot_iter in [40, 80, 120]:
            first_snapshot = os.path.join(snapshot_dir, "snapshots/boby_iter_{}.pth".format(snapshot_iter))
            self.assertTrue(os.path.exists(first_snapshot))

    def _test_train(self, m: nn.Module) -> None:
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        mock_optimizer = mock.MagicMock(spec_set=optimizer)
        mock_optimizer.param_groups = optimizer.param_groups
        mock_optimizer.zero_grad.__signature__ = inspect.signature(optimizer.zero_grad)
        mock_optimizer.state_dict.return_value = {}

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=mock_optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)
        self.assertTrue(m.training)
        self.assertEqual(m.count, 45)
        mock_optimizer.assert_has_calls([call.zero_grad(set_to_none=True)] * 45, any_order=True)
        mock_optimizer.assert_has_calls([call.step()] * 45, any_order=True)

    def test_train(self) -> None:
        self._test_train(m=FakeModel())

    def test_train_loss_components_dict(self) -> None:
        self._test_train(m=FakeModel2())

    def test_train_loss_components_list(self) -> None:
        self._test_train(m=FakeModel3())

    def test_train_fusedadam(self) -> None:
        m = FakeModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        mock_optimizer = mock.MagicMock(spec_set=optimizer)
        mock_optimizer.param_groups = optimizer.param_groups
        mock_optimizer.zero_grad.__signature__ = inspect.signature(optimizer.zero_grad).replace(parameters=[])
        mock_optimizer.state_dict.return_value = {}

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=mock_optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)
        self.assertTrue(m.training)
        self.assertEqual(m.count, 45)
        mock_optimizer.assert_has_calls([call.zero_grad()] * 45, any_order=True)
        mock_optimizer.assert_has_calls([call.step()] * 45, any_order=True)

    @pytest.mark.gpu
    @unittest.skipIf(not torch.cuda.is_available(), "no gpu installed, skipping test")
    def test_gpu_train(self) -> None:
        m = FakeModel()
        m.cuda()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        mock_optimizer = mock.MagicMock(spec_set=optimizer)
        mock_optimizer.param_groups = optimizer.param_groups
        mock_optimizer.zero_grad.__signature__ = inspect.signature(optimizer.zero_grad)
        mock_optimizer.state_dict.return_value = {}

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=mock_optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)
        self.assertTrue(m.training)
        self.assertEqual(m.count, 45)
        mock_optimizer.assert_has_calls([call.zero_grad(set_to_none=True)] * 45, any_order=True)
        mock_optimizer.assert_has_calls([call.step()] * 45, any_order=True)

    @pytest.mark.gpu
    @unittest.skipIf(not torch.cuda.is_available(), "no gpu installed, skipping test")
    def test_gpu_train_mixed_precision_fp16(self) -> None:
        m = FakeModel()
        m.cuda()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            precision=torch.float16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)
        self.assertTrue(m.training)
        self.assertEqual(m.count, 45)

    @pytest.mark.gpu
    @unittest.skipIf(not torch.cuda.is_available(), "no gpu installed, skipping test")
    def test_gpu_train_mixed_precision_bf16(self) -> None:
        m = FakeModel()
        m.cuda()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            precision=torch.bfloat16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        solver.execute([0, 1], run_dir)
        self.assertTrue(m.training)
        self.assertEqual(m.count, 45)

    def test_nan_loss_debug(self) -> None:
        m = FakeModel4()
        optimizer = torch.optim.Adam(m.parameters(), lr=0.78)
        lr_scheduler = get_constant_schedule(optimizer)

        mock_optimizer = mock.MagicMock(spec_set=optimizer)
        mock_optimizer.param_groups = optimizer.param_groups
        mock_optimizer.zero_grad.__signature__ = inspect.signature(optimizer.zero_grad)
        mock_optimizer.state_dict.return_value = {}

        m.eval()  # test that solver moves model to train
        solver = BasicSolver(
            model=m,
            optimizer=mock_optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=16,
            train_dataset_loader=FakeDataLoader(list(np.arange(100 * 4).reshape(100, 4))),
            model_forward_func=lambda m, b: m(b),
            iterations=45,
        )
        run_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

        with self.assertRaises(ValueError) as e:
            solver.execute([0, 1], run_dir)
        self.assertEqual(str(e.exception), "Training loss cannot be Nan.")


if __name__ == "__main__":
    unittest.main()
