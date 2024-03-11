import socket
import unittest
from unittest import mock

import pytest
import torch
from omnisearchsage.modules.negatives import AllGatherEmbedding
from omnisearchsage.modules.negatives import AllGatherWithGrad
from torch.autograd import gradcheck
from torch.cuda import set_device


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return str(s.getsockname()[1])


def all_gather_example(x):
    embed = AllGatherEmbedding.apply(x)
    return torch.sum(embed)


class AllGatherEmbeddingTest(unittest.TestCase):
    def test_single_cpu(self):
        input = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(all_gather_example, input, eps=1e-6, atol=1e-4))

    def test_requires_grad(self):
        input = torch.randn(2, 4, dtype=torch.double, requires_grad=False)
        out = AllGatherEmbedding.apply(input)
        self.assertFalse(out.requires_grad)


class AllGatherWithGradTest(unittest.TestCase):
    @staticmethod
    def setup_ddp(rank, world_size):
        """Setup ddp enviroment."""
        set_device(rank)
        # gloo/mpi/nccl don't support reduce_scatter on cpu
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    @staticmethod
    def _all_gather_ddp_test_helper_1d(rank, world_size):
        AllGatherWithGradTest.setup_ddp(rank, world_size)

        tensor = torch.ones((2 + rank,), requires_grad=True, device=rank)
        # [1, 1], [1, 1, 1], [1, 1, 1, 1]
        gathered = AllGatherWithGrad.apply(tensor * (rank + 1.0))
        for ix, v in enumerate(gathered):
            assert v.shape == (2 + ix,), (v.shape, ix)
        # [1, 1, 2, 2, 2, 3, 3, 3, 3]
        gathered = torch.cat(gathered, dim=0)
        # scales by 1, 2, 3 -> total scale of gradient == 6
        gathered = gathered * (rank + 1)
        gathered.sum().backward()

        grad1 = torch.zeros_like(tensor).fill_(torch.arange(1, world_size + 1).sum().float() * (rank + 1.0))

        assert torch.allclose(grad1, tensor.grad)

    @staticmethod
    def _all_gather_ddp_test_helper_2d(rank, world_size):
        AllGatherWithGradTest.setup_ddp(rank, world_size)

        tensor = torch.ones((rank + 2, rank + 2), requires_grad=True, device=rank)
        gathered = AllGatherWithGrad.apply(tensor * (rank + 1.0))
        gathered = torch.cat([t.view(-1) for t in gathered], dim=0) * (rank + 1)
        gathered.sum().backward()

        grad1 = torch.zeros_like(tensor).fill_(torch.arange(1, world_size + 1).sum().float() * (rank + 1.0))

        assert torch.allclose(grad1, tensor.grad)

    @staticmethod
    def _all_gather_ddp_test_helper_0d(rank, world_size):
        AllGatherWithGradTest.setup_ddp(rank, world_size)

        tensor = torch.ones((), requires_grad=True, device=rank)
        gathered = AllGatherWithGrad.apply(tensor * (rank + 1.0))
        gathered = torch.cat([t.view(-1) for t in gathered], dim=0) * (rank + 1)
        gathered.sum().backward()

        grad1 = torch.zeros_like(tensor).fill_(torch.arange(1, world_size + 1).sum().float() * (rank + 1.0))

        assert torch.allclose(grad1, tensor.grad)

    @pytest.mark.gpu
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        "need at least 2 gpus to test collective ops",
    )
    @mock.patch.dict("os.environ", {"MASTER_ADDR": "localhost", "MASTER_PORT": find_free_port()})
    def test_all_gather_ddp_0d(self):
        world_size = 2
        torch.multiprocessing.spawn(self._all_gather_ddp_test_helper_0d, args=(world_size,), nprocs=world_size)

    @pytest.mark.gpu
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        "need at least 2 gpus to test collective ops",
    )
    @mock.patch.dict("os.environ", {"MASTER_ADDR": "localhost", "MASTER_PORT": find_free_port()})
    def test_all_gather_ddp_1d(self):
        world_size = 2
        torch.multiprocessing.spawn(self._all_gather_ddp_test_helper_1d, args=(world_size,), nprocs=world_size)

    @pytest.mark.gpu
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        "need at least 2 gpus to test collective ops",
    )
    @mock.patch.dict("os.environ", {"MASTER_ADDR": "localhost", "MASTER_PORT": find_free_port()})
    def test_all_gather_ddp_2d(self):
        world_size = 2
        torch.multiprocessing.spawn(self._all_gather_ddp_test_helper_2d, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    unittest.main()
