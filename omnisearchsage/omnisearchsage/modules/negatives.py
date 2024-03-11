from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import warnings
from functools import reduce

import torch
import torch.nn.functional as F
from torch.distributed import all_gather
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.distributed import group


def all_gather_1d_tensor(
    tensor: torch.Tensor, group=group.WORLD, expect_equal_sizes: bool = False
) -> List[torch.Tensor]:
    """
    Gathers variable size dimension 1 tensors into a list from all processes in group

    Args:
        tensor: tensor to gather
        group: group to gather over
        expect_equal_sizes: if True, throws an error if all tensors are not the same size

    Returns:
        List of tensors from each process in group
    """
    if not torch.distributed.is_initialized():
        warnings.warn("Only can gather in non-distributed setting. Only returning input tensor")
        return [tensor]
    if tensor.ndim != 1:
        raise ValueError(f"Only can gather variable size tensors with ndim==1. found shape {tensor.shape}")
    if not tensor.is_cuda:
        raise ValueError(
            f"Required tensor on device equal to rank. Found: device={tensor.device}, cur_rank={get_rank()}"
        )
    world_size = get_world_size()

    cur_dev = tensor.device

    # first gather shapes and find largest input
    cur_shape = torch.tensor([tensor.size(0)], device=cur_dev)
    shapes_t = torch.empty(world_size, dtype=torch.long, device=cur_dev)
    shapes_l = [shapes_t[i : i + 1] for i in range(world_size)]
    all_gather(tensor_list=shapes_l, tensor=cur_shape, group=group)
    max_size = shapes_t.max().item()
    if expect_equal_sizes:
        min_size = shapes_t.min().item()
        if min_size != max_size:
            raise ValueError(f"Found different sizes in allreduce. shapes={shapes_t.tolist()}")
    # then gather tensors themselves. Because all_gather requires equal size tensors in all ranks,
    # we resize the tensors to the size of the longest one, then prune the newly added bits after the gather
    values_t = torch.empty((world_size * max_size,), device=cur_dev, dtype=tensor.dtype)
    values_l = [values_t[i * max_size : (i + 1) * max_size] for i in range(world_size)]
    all_gather(tensor_list=values_l, tensor=F.pad(tensor.reshape(-1), (0, max_size - tensor.numel())), group=group)
    return [values_l[i][: shapes_t[i]] for i in range(world_size)]


class AllGatherEmbedding(torch.autograd.Function):
    """
    No-op if torch.distributed.is_initialized() is False.

    Custom autograd Function to all gather embeddings from every GPU process.

    Given M processes each with a (N x D) embedding, we all gather embeddings from
    every process so each of the M processes result in the same (M * N x D) embedding
    """

    @staticmethod
    def forward(ctx, embedding: torch.Tensor) -> torch.Tensor:
        """
        Use torch.distributed.all_gather to communicate between multiple processes
        to send embeddings. By default, all_gather is not auto-grad aware so we
        need to define our own custom backwards function.
        """

        if not torch.distributed.is_initialized():
            return embedding

        ctx.embed_size = embedding.size(0)
        embed_size_tensors = list(
            torch.zeros(torch.distributed.get_world_size(), 1, dtype=torch.int32, device=embedding.device).unbind(0)
        )

        embed_size_tensor = torch.tensor([ctx.embed_size], dtype=torch.int32, device=embedding.device)
        # All gather output ordered by rank index
        # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce

        torch.distributed.all_gather(embed_size_tensors, embed_size_tensor)
        for emb_size in embed_size_tensors:
            # we can probably get around this requirement but its tricky enough to not do so for now
            assert ctx.embed_size == emb_size, "Encountered different tensor sizes in all reduce. {} vs {}".format(
                ctx.embed_size, emb_size
            )

        all_embedding = list(
            torch.zeros(
                torch.distributed.get_world_size(),
                embedding.size(0),
                embedding.size(1),
                dtype=embedding.dtype,
                device=embedding.device,
            ).unbind(0)
        )
        torch.distributed.all_gather(all_embedding, embedding.contiguous())

        all_embedding = torch.cat(all_embedding, dim=0)
        all_embedding.requires_grad = embedding.requires_grad

        return all_embedding

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        if not torch.distributed.is_initialized():
            return grad_output

        rank = torch.distributed.get_rank()

        torch.distributed.all_reduce(grad_output.contiguous(), torch.distributed.ReduceOp.SUM)
        return grad_output[rank * ctx.embed_size : (rank + 1) * ctx.embed_size, :]


def _int_product(nums: Iterable[int]) -> int:
    return reduce(lambda a, b: a * b, nums, 1)


class AllGatherWithGrad(torch.autograd.Function):
    """
    No-op if torch.distributed.is_initialized() is False.

    Custom autograd Function to all gather tensors from every GPU process (that supports
    backpropagation)

    As opposed to AllGatherEmbedding, there is no requirement that all processes have the same
    input size, only the same input number of dimensions (this assumption is not required, but saves
    an additional all_gather)
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional[torch.distributed.ProcessGroup] = group.WORLD,
    ) -> Tuple[torch.Tensor, ...]:
        if not torch.distributed.is_initialized():
            return (tensor,)

        ctx.group = group
        ctx.shape = tensor.shape
        ctx.dtype = tensor.dtype
        ctx.device = tensor.device

        embed_sizes = torch.zeros(
            get_world_size(group),
            tensor.ndim,
            dtype=torch.int32,
            device=tensor.device,
        )
        # Note: assumes ndim is same in all processes
        cur_embed_size = torch.tensor(list(tensor.shape), dtype=torch.int32, device=tensor.device)
        torch.distributed.all_gather(
            tensor_list=list(embed_sizes.unbind(0)),
            tensor=cur_embed_size,
            group=group,
        )
        # if we gathered embeddings using tensors of shape cur_embed_size.max(0)
        # it would be at least as large as max_size, so instead we reshape to dim 1 then gather
        max_size = embed_sizes.prod(1).max().item()
        ctx.max_size = max_size

        raw_gathered = torch.empty(get_world_size(group), max_size, device=tensor.device, dtype=tensor.dtype)

        tensor_padded = F.pad(tensor.reshape(-1), (0, max_size - tensor.numel()))

        torch.distributed.all_gather(
            tensor_list=list(raw_gathered.unbind(0)),
            tensor=tensor_padded,
            group=group,
        )
        gathered_tensors = []
        for ix, shape in enumerate(embed_sizes.tolist()):
            if shape:
                gathered_tensors.append(raw_gathered[ix, : _int_product(shape)].view(*shape))
            else:
                gathered_tensors.append(raw_gathered[ix, 0])

        return tuple(gathered_tensors)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        if not torch.distributed.is_initialized():
            return grad_output[0], None
        grads_to_scatter = [F.pad(o.view(-1), (0, ctx.max_size - o.numel())) for o in grad_output]
        output_tensor = torch.empty(ctx.max_size, dtype=ctx.dtype, device=ctx.device)
        torch.distributed.reduce_scatter(
            output=output_tensor,
            input_list=grads_to_scatter,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False,
            group=ctx.group,
        )
        if ctx.shape:
            valid_subset = output_tensor[: _int_product(ctx.shape)].view(*ctx.shape)
        else:
            valid_subset = output_tensor[0]
        return valid_subset, None
