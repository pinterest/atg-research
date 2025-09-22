from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Iterable
import logging
from functools import reduce

import re
import sys
import math
import random
from torch import nn
from xml.dom import minidom
from torch.distributed import group
from torch.distributed import all_gather
from torch.distributed import get_rank
from torch.distributed import get_world_size

# import boto3
import numpy as np
import torch
from typing import Tuple
import torch.nn.functional as F
LOG = logging.getLogger(__name__)
# from botocore.exceptions import ClientError
# from botocore.exceptions import NoCredentialsError

# for two-tower model flags
IMG_SIG_TAG = "sig"
ITEM_ID_TAG = "LABEL/candidate_signature"  # use signature for now as item id is absent

if TYPE_CHECKING:
    from torch import nn
    from interactrank.common.utils.utils import ExampleType

DUPLICATE_METHODS = ("average", "representative", "none")
BASE = sys.maxsize

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
class InBatchNegativeLossMultihead(nn.Module):
    """
    Calculate in-batch negative loss given viewer embeddings and item embeddings
    If initialized with an item_counter that is not None, will perform sample probability correction.
    Use matrix operation to replace the for loop iteration to speed up
    """

    def __init__(
        self,
        temperature_scale: float = 1.0,
        item_counter: Counter = None,
    ):
        """
        :param temperature_scale: a float represent the temperature scaling for softmax
        :param item_counter: None or an instance of CpuBasedCounter
        :param dedup_method: One of 'average, 'representative', or 'none'
        """
        super().__init__()
        self.temperature_scale = nn.Parameter(torch.tensor(-math.log(temperature_scale)))
        self.criterion = nn.BCELoss(reduction="none")
        self.item_counter = item_counter
        self.correct_sample_probability = item_counter is not None

    def forward(
        self,
        viewer_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        true_label_mask: torch.Tensor,
        weights: torch.Tensor,
        items: torch.Tensor,
        negative_items: torch.Tensor,
        unique_neg_embeddings: torch.Tensor,
        unique_neg_items: torch.Tensor,
    ):
        """
        calculate in batch negative loss given a batch of
        viewer embedding and item embeddings and a set of negative
        items.

        :param viewer_embeddings: Batch viewer embeddings [B, H, D], grad, float32
        :param entity_embeddings: Batch pin embeddings [B, D], grad, float32
        :param true_label_mask: a torch Tensor of shape: [H, B], nograd, bool
        :param weights: weights of each training sample [H, B], nograd, float32
        :param items: 1-D tensor of item (img sig) ids [B], nograd, int64
        :param negative_items: [B, embedding_size], nograd, int64
        :param unique_neg_embeddings: [B, embedding_size], grad, float32
        :param unique_neg_items: [B,], nograd, int64

        :return: a torch Tensor represent the in batch negative loss
        """

        # Get all the dimensions and shapes
        num_heads = viewer_embeddings.shape[1]  # H
        batch_size = viewer_embeddings.shape[0]  # B
        device = viewer_embeddings.device

        # Advanced indexing
        # We are computing in batch negatives for each head stacking together
        # We make use of the linearity and non-interactive nature of each register,
        # a (task, entry) pair where true_label_mask[task, entry] is True
        true_label_idx = torch.arange(batch_size, dtype=torch.int64, device=device)
        valid_register = torch.cat([true_label_idx[true_label_mask[i]] for i in range(num_heads)], dim=0)
        # breakpoints = true_label_mask.sum(dim=0).cumsum(dim=0)
        in_batch_neg_per_sample_weights = torch.cat([weights[i, true_label_mask[i]] for i in range(num_heads)], dim=0)

        # We have to swap axis [batch, task, embedding] to [task, batch, embedding]
        # because we register with (task, entry) pair for true_label_idx and valid_register
        pos_viewer_embeddings = viewer_embeddings.swapaxes(0, 1).reshape(-1, viewer_embeddings.shape[-1])[
            true_label_mask.flatten()
        ]

        # below is for calculating in batch negative loss
        self.temperature_scale.data.clamp_(max=math.log(100.0))

        pos_entity_embeddings = torch.cat([entity_embeddings[true_label_mask[i]] for i in range(num_heads)], dim=0)
        positive_logits = torch.sum(pos_viewer_embeddings * pos_entity_embeddings, dim=-1) * torch.exp(
            self.temperature_scale
        )  # [S]

        negative_logits = (pos_viewer_embeddings @ unique_neg_embeddings.T) * torch.exp(
            self.temperature_scale
        )  # [S, B]
        positive_item_eq_negative = torch.eq(items.reshape(-1, 1), unique_neg_items.reshape(1, -1))  # [B, B]
        negative_logits.masked_fill_(positive_item_eq_negative[valid_register], float("-inf"))

        if self.correct_sample_probability:
            if len(valid_register) < batch_size:
                pos_counts, total = self.item_counter(items[valid_register])
                pos_probs = (pos_counts / total).type(torch.float64)
                pos_batch_probs = (1.0 - (1.0 - pos_probs) ** negative_items.numel()).type(torch.float32)  # [B]
                # [S] = [S] - [S]
                corrected_positive_logits = positive_logits - torch.log(torch.clamp(pos_batch_probs, 1e-16, 1.0))
            else:
                pos_counts, total = self.item_counter(items)
                pos_probs = (pos_counts / total).type(torch.float64)
                pos_batch_probs = (1.0 - (1.0 - pos_probs) ** negative_items.numel()).type(torch.float32)  # [B]
                # [S] = [S] - [S]
                corrected_positive_logits = (
                    positive_logits - torch.log(torch.clamp(pos_batch_probs, 1e-16, 1.0))[valid_register]
                )

            neg_counts, total = self.item_counter(unique_neg_items)
            neg_probs = (neg_counts / total).type(torch.float64)
            neg_batch_probs = (1.0 - (1.0 - neg_probs) ** negative_items.numel()).type(torch.float32)
            # [S, B] = [S, B] - [B]
            corrected_negative_logits = negative_logits - torch.log(torch.clamp(neg_batch_probs, 1e-16, 1.0))
        else:
            corrected_positive_logits, corrected_negative_logits = positive_logits, negative_logits

        softmax_logits = torch.cat(
            [corrected_positive_logits.reshape(-1, 1), corrected_negative_logits], dim=1
        )  # [S, B+1]
        # Softmax creates some floating point error
        in_batch_neg_pred = F.softmax(softmax_logits, dim=-1)
        # true_label_preds: [NUM_USER]
        true_label_preds = in_batch_neg_pred[:, 0]  # We only need the first one, but it is faster to do this way on GPU

        # in_batch_loss_per_sample: [NUM_USER]
        in_batch_loss_per_sample = self.criterion(true_label_preds, torch.ones_like(true_label_preds))  # [B]
        in_batch_loss = torch.sum(in_batch_loss_per_sample * in_batch_neg_per_sample_weights) / num_heads
        logit_matrix = torch.cat([positive_logits.reshape(-1, 1), negative_logits], dim=1)
        logit_matrix_list = logit_matrix.split(true_label_mask.sum(-1).tolist())

        return in_batch_loss, None, logit_matrix_list

class Counter(nn.Module):
    def update(self, longs: torch.Tensor, increment: int = 1) -> None:
        raise NotImplementedError()

    def forward(self, longs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class CountMinSketch(Counter):
    def extra_repr(self) -> str:
        return f"w={self.w},d={self.d}"

    def __init__(self, w: int, d: int, seed: Optional[int] = None, synchronize_counts: bool = False):
        """
        Implements count-min sketch, expecting inputs that are of dtype int64
        Args:
            w: width of sketch
            d: depth of sketch
            seed: random seed to use for hash function initialization. must be specified and the same across all
                processes if synchronize_counts=True
            synchronize_counts: if True, gathers inputs from all processes in `update`
        """
        super().__init__()
        if synchronize_counts and (seed is None):
            raise ValueError("if synchronize_counts is True, seed must be set")
        self.synchronize_counts = synchronize_counts
        self.w = w
        self.d = d
        LOG.info(f"initializing count-min sketch with width={self.w}, depth={self.d}")
        self.register_buffer("counts", torch.zeros((self.d, self.w), dtype=torch.long))
        self.register_buffer("hash_a", torch.zeros(self.d, dtype=torch.long))
        self.register_buffer("idx", torch.arange(self.d))
        self.register_buffer("num_seen", torch.tensor(0))

        r = random.Random(seed)
        for i in range(self.d):
            self.hash_a[i] = r.randint(1, torch.iinfo(torch.long).max)

    @torch.jit.script
    def hash_func(longs: torch.Tensor, w: int, hash_a: torch.Tensor) -> torch.Tensor:
        # shape: (...) -> (..., self.d)
        PRIME_MODULUS = (1 << 31) - 1
        hash_ = longs.unsqueeze(-1) * hash_a
        hash_ += hash_ >> 32
        hash_ &= PRIME_MODULUS
        return (hash_.int() % w).long()

    def _hash(self, longs: torch.Tensor) -> torch.Tensor:
        return self.hash_func(longs=longs, w=self.w, hash_a=self.hash_a)

    def all_gather_1d_tensor(self,
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
        shapes_l = [shapes_t[i: i + 1] for i in range(world_size)]
        torch.distributed.all_gather(tensor_list=shapes_l, tensor=cur_shape, group=group)
        max_size = shapes_t.max().item()
        if expect_equal_sizes:
            min_size = shapes_t.min().item()
            if min_size != max_size:
                raise ValueError(f"Found different sizes in allreduce. shapes={shapes_t.tolist()}")
        # then gather tensors themselves. Because all_gather requires equal size tensors in all ranks,
        # we resize the tensors to the size of the longest one, then prune the newly added bits after the gather
        values_t = torch.empty((world_size * max_size,), device=cur_dev, dtype=tensor.dtype)
        values_l = [values_t[i * max_size: (i + 1) * max_size] for i in range(world_size)]
        torch.distributed.all_gather(tensor_list=values_l, tensor=F.pad(tensor.reshape(-1), (0, max_size - tensor.numel())), group=group)
        return [values_l[i][: shapes_t[i]] for i in range(world_size)]

    def update(self, longs: torch.Tensor, increment: int = 1):
        """
        Increments the ids in `longs` by `increment` inside the sketch
        """
        # assert self.training
        hashes = self._hash(longs.view(-1))  # (product(longs.shape), self.d)
        if self.synchronize_counts:
            assert hashes.ndim == 2, hashes.shape
            hashes = torch.cat(self.all_gather_1d_tensor(hashes.view(-1)), dim=0).view(-1, self.d)
        self.counts.index_put_((self.idx, hashes), torch.full_like(hashes, fill_value=increment), accumulate=True)
        self.num_seen += hashes.size(0) * increment

    @torch.no_grad()
    def forward(self, longs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hashes = self._hash(longs)
        min_cts = self.counts[self.idx, hashes].min(-1).values
        return min_cts, self.num_seen

@torch.no_grad()
def batch_embedding_stats_analysis(tensor):
    """
    input: tensor of shape [batch, embedding]
    output: dict of the statistics that can be computed within the tensor
        empty_rate: the percentage of all-zero embeddings
        sparsity_rate: the percentage of zero entries
        mean_norm: the average norm of the tensor (for normalized tensors, this should always be 1)
        variance: variance in between the tensors
    """
    tol = 1e-5
    assert tensor.dim() == 2
    stats = dict()

    # Entry-level statistics
    stats["empty_rate"] = (tensor.norm(dim=-1) < tol).float().mean()
    stats["sparsity_rate"] = (tensor.abs() < tol).float().mean()
    stats["mean_norm"] = tensor.norm(dim=-1).mean()

    # Batch-level statistics
    ex = tensor.mean(dim=0)
    e_x2 = (tensor * tensor).mean(dim=0)
    var = (e_x2 - ex**2).sum()
    stats["variance"] = var

    return {key: stats[key].cpu().numpy().tolist() for key in stats}

def _stack_sigs(batch: ExampleType) -> ExampleType:
    if IMG_SIG_TAG in batch:
        batch[IMG_SIG_TAG] = torch.stack(batch[IMG_SIG_TAG], dim=0)
    return batch


def get_recall_metrics(logit_matrix: torch.Tensor, ks: List[int]):
    """
    Get recall @ k metrics given the logit matrix.  The logit matrix is assumend to be formatted so that the
    true label logits appear in the first column and negative label logits in all remaining columns.

    :param logit_matrix: tensor of shape [NUM_USER, BATCH_SIZE + 1], e.g [1000, 6001]
    :param ks: list of k values.
    :return: map of recall metrics keyed by k.
    """
    with torch.no_grad():
        num_item = logit_matrix.shape[0]
        # true_label_logits: (num_item,)
        true_label_logits = logit_matrix[:, 0]

        # ranks: [num_item]
        ranks = torch.sum((logit_matrix[:, 1:] >= true_label_logits.reshape(-1, 1)).float(), dim=1)
        return {k: (torch.sum(ranks < k) / num_item) for k in ks}


def get_embedding_norm_metric(embeddings: torch.Tensor):
    """
    get embedding norm related metrics given unnormalized embeddings

    :param embeddings: tensor of shape [BATCH_SIZE, EMBEDDING_DIM], e.g [6000, 64]
    :return: maximum norm metrics
    """
    with torch.no_grad():
        return torch.max(torch.linalg.norm(embeddings, dim=1)).detach()


def scale_gradient(value: torch.Tensor, scale: float):
    """
    scale the gradient of the input value with float scale

    :param value:  the value to scale
    :param scale:  scale
    :return: the scaled torch tensor
    """
    with torch.no_grad():
        scaled_value = value * (1.0 - scale)
    return value * scale + scaled_value


def convert_img_sig_tensor_to_long(tensor: torch.Tensor) -> torch.Tensor:
    """
    generate long integer for (N, 32) int8 img sig tensor; for example one
    tensor([55,  56,  53,  97,  51,  51,  53,  54,  57,  51,  49,  54,  99,  57,
         49,  51,  54,  97,  52,  99,  50,  49,  54,  56,  56,  55,  97,  48,
         98,  98, 102,  98], dtype=torch.int8) represents img sig 785a33569316c9136a4c216887a0bbfb
    :param tensor: (N, 32) int8 tensor where each row represents an img sig
    :return: (N, 1) long integer tensor where each value represent a new long id for an img sig
    """
    # transfer the tensor value to [0, 15]
    tensor = (tensor - 48).masked_fill(tensor - 48 > 10, 0) + (tensor - 97 + 10).masked_fill(tensor - 97 < 0, 0)
    weight = torch.as_tensor(2 ** torch.arange(0, 64, 2), device=tensor.device)
    return (tensor * weight).sum(1).long()

def unique_indices_1d(items: torch.Tensor):
    """
    get the indices of the unique elements in items; for tensor([1, 1, 2, 1]), it will return tensor([0, 2])
    :param items: 1-D tensor
    :return: indices of the unique elements
    """
    values, indices = torch.sort(items, stable=True)
    unique_mask = torch.cat([torch.tensor([True], device=items.device), values[1:] > values[:-1]], dim=-1)
    return indices[unique_mask]


def unique_index_inverse_and_counts(items: torch.Tensor):
    """
    get unique elements, their indices, inverse, and counts;
    same as np.unique(items, return_index=True, return_inverse=True, return_counts=True)
    :param items: 1-D tensor
    :return: unique elements, indices, inverse, and counts
    """
    unique, inverse, counts = torch.unique(items, return_inverse=True, return_counts=True)
    indices = unique_indices_1d(inverse)
    return unique, indices, inverse, counts


def uniquify_with_average(items, embeddings):
    """
    Dedup the embeddings.  The resulting embedding for an item will be the average of that item's embeddings.
    :param items: 1-D tensor of item IDs [N]
    :param embeddings: [N, D] tensor of item embeddings, which should be the same size as items in the first dimension.
    :return: unique item IDs, their embeddings
    """
    unique, _, inverse, counts = unique_index_inverse_and_counts(items)
    num_unique = unique.shape[0]
    embedding_dim = embeddings.shape[1]
    result = torch.zeros(num_unique, embedding_dim, device=embeddings.device)
    result.scatter_add_(0, inverse.reshape(-1, 1).tile(1, embedding_dim), embeddings)
    return unique, result / counts.reshape(-1, 1)


def uniquify_embeddings(items, embeddings):
    """
    Dedup the embeddings.  The resulting embedding for an item will one of the item's embeddings chosen randomly.
    :param items: 1-D tensor of item IDs [N]
    :param embeddings: [N, D] tensor of item embeddings, which should be the same size as items in the first dimension.
    :return: unique item IDs, their embeddings
    """
    unique, index, _, _ = unique_index_inverse_and_counts(items)
    return unique, embeddings[index]


def correct_sample_probability(logits, label_prob, true_label_mask, negative_indices):
    """
    Adjusts the given logits based on the sample probability of the labels.
    :param logits: results returned from deduped_batch_logits
    :param label_prob: probability of each label (item) Shape [B,]
    :param true_label_mask: a boolean vector that identifies the users with true labels [B,]
    :param negative_indices: the indices of negative labels
    :return: the adjusted logit matrix
    """
    true_label_prob = torch.masked_select(label_prob, true_label_mask).type(torch.float64)
    true_batch_prob = (1.0 - (1.0 - true_label_prob) ** label_prob.numel()).type(torch.float32)
    negative_label_prob = torch.index_select(label_prob, 0, negative_indices).type(torch.float64)
    negative_batch_prob = (1.0 - (1.0 - negative_label_prob) ** label_prob.numel()).type(torch.float32)
    tiled_neg_batch_prod = torch.tile(negative_batch_prob.reshape(1, -1), [true_batch_prob.numel(), 1])
    prob_matrix = torch.cat((true_batch_prob.reshape(-1, 1), tiled_neg_batch_prod), -1)
    return logits - torch.log(torch.clamp(prob_matrix, 1e-16, 1.0))


def softmax_loss_over_impressed_negatives(
    in_batch_logits: torch.Tensor,
    user_ids: torch.Tensor,
    true_label_mask: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
):
    """
    Calculate the softmax loss over only items from the same user.
    Identify the same user mask with the user ids read from the data.

    :param in_batch_logits:  a torch Tensor of shape: [NUM_USER, BATCH_SIZE]
    :param user_ids: a torch Tensor of shape: [NUM_USER]
    :param true_label_mask: a torch Tensor of shape: [BATCH_SIZE]
    :param criterion: the criterion used for calculating the loss
    :return a torch Tensor indicating the impressed softmax loss
    """
    num_user = in_batch_logits.shape[0]
    # get unique user ids, shape [NUM_USER]
    unique_users = user_ids[true_label_mask]
    # same_user_mask: shape [NUM_USER, BATCH_SIZE], boolean tensor
    same_user_mask = torch.eq(torch.unsqueeze(unique_users, -1), torch.unsqueeze(user_ids, 0))
    # masked_logits: shape [NUM_USER, BATCH_SIZE], float tensor
    # masked all non-same user-item entry as -inf
    masked_logits = torch.where(same_user_mask, in_batch_logits, torch.tensor(float("-inf")))

    # masked_softmax: shape [NUM_USER, BATCH_SIZE], float tensor
    # all non-same user-item entry as 0.0
    masked_softmax = F.softmax(masked_logits, dim=-1)
    # indices of all the true prediction entries, shape [NUM_USER, 2]
    indices = torch.stack([torch.arange(num_user), torch.where(true_label_mask)[0]], dim=-1)
    # true_preds [NUM_USER]. indicate all the true prediction of user-item pair (from the same user)
    true_preds = torch.gather(masked_softmax, dim=1, index=indices)[:, 1]
    loss = criterion(true_preds, torch.ones_like(true_preds))
    return loss


def in_batch_negative_loss(
    in_batch_neg_logits: torch.Tensor,
    true_label_mask: torch.Tensor,
    weights: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
):
    """
    calculate in batch negative loss given the in batch negative logits
    and label mask
    :param in_batch_neg_logits: Batch pin embeddings [NUM_USER, BATCH_SIZE]
    :param true_label_mask: a torch Tensor of shape: [BATCH_SIZE]
    :param weights: Batch pin embeddings [BatchSize]
    :param criterion: an instance of nn.BCEWithLogitsLoss criterion
    :return: a torch Tensor represent the in batch negative loss
    """

    # in_batch_neg_pred: [NUM_USER, BATCH_SIZE]
    in_batch_neg_pred = F.softmax(in_batch_neg_logits, dim=-1)
    # in_batch_neg_per_sample_weights: [NUM_USER]
    in_batch_neg_per_sample_weights = weights[true_label_mask]

    # true_label_preds: [NUM_USER]
    true_label_preds = torch.diagonal(in_batch_neg_pred[:, true_label_mask])

    # in_batch_loss_per_sample: [NUM_USER]
    in_batch_loss_per_sample = criterion(true_label_preds, torch.ones_like(true_label_preds))
    in_batch_loss = torch.sum(in_batch_loss_per_sample * in_batch_neg_per_sample_weights)
    return in_batch_loss


def maybe_dedup_negatives(
    negative_embeddings: torch.Tensor,
    negative_items: torch.Tensor,
    dedup_method: str = "none",
):
    """

    :param negative_embeddings: torch Tensor of shape [B,
    :param negative_items: torch Tensor of shape
    :param dedup_method: one of average, representative, none

    :return: unique_neg_items, unique_neg_embeddings
    """
    if dedup_method not in DUPLICATE_METHODS:
        raise ValueError(f"dedup_method should be one of {DUPLICATE_METHODS}, not '{dedup_method}'")

    if dedup_method == "average":
        unique_neg_items, unique_neg_embeddings = uniquify_with_average(negative_items, negative_embeddings)
    elif dedup_method == "representative":
        unique_neg_items, unique_neg_embeddings = uniquify_embeddings(negative_items, negative_embeddings)
    elif dedup_method == "none":
        unique_neg_items, unique_neg_embeddings = negative_items, negative_embeddings
    else:
        assert False, f"dedup_method should not be '{dedup_method}'"

    return unique_neg_items, unique_neg_embeddings

def get_partition_identifier(path, db_name):
    # Use regular expression to find 'adhoc_dataset_name' and its value in the path
    adhoc_dataset_name_match = re.search(r"(adhoc_dataset_name=[^/]*)", path)
    # If 'adhoc_dataset_name' is found, append a '/' at the end for formatting; if not, return an empty string
    adhoc_dataset = (adhoc_dataset_name_match.group(1) + "/") if adhoc_dataset_name_match is not None else ""

    # Use regular expression to find the 'dt' date part, which is expected to be in the format 'dt=YYYY-MM-DD'
    # Example Output: "dt=2024-04-29"
    dt_id = re.search(r"/(dt=[^/]*)", path).group(1).replace("_", "-")
    table_name = re.search("{}/([^/]*)".format(db_name), path).group(1)

    # Concatenate adhoc_dataset, dt identifier with hyphens, and table name separated by a colon to form the partition identifier
    partition_identifier = adhoc_dataset + dt_id + f":{table_name}"
    return partition_identifier
