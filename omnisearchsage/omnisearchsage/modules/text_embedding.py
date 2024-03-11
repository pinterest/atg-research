from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing_extensions import Final

import math
import random

import torch
from omnisearchsage.modules.sage_embedder import SageEmbedder
from torch import nn
from transformers import AutoModel

if TYPE_CHECKING:
    from torch import Tensor


class LpNormalize(nn.Module):
    def __init__(self, p: float = 2.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=self.p, dim=-1)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class EmbeddingPrecisionLayer(nn.Module):
    def __init__(self, precision: torch.dtype) -> None:
        super().__init__()
        self.precision = precision

    def forward(self, x: Tensor) -> Tensor:
        return x.to(self.precision)

    def extra_repr(self) -> str:
        return f"precision={self.precision}"


def _generate_mlp(
    *dims: int, layernorm: bool = False, normalize: bool = False, precision: Optional[torch.dtype] = None
) -> nn.Module:
    modules = []
    for input_dim, output_dim in zip(dims[:-2], dims[1:-1]):
        modules.append(nn.Linear(input_dim, output_dim))
        modules.append(nn.ReLU(inplace=True))
        if layernorm:
            modules.append(nn.LayerNorm(output_dim))
    if len(dims) >= 2:
        modules.append(nn.Linear(dims[-2], dims[-1]))
    if precision is not None:
        modules.append(EmbeddingPrecisionLayer(precision))
    if normalize:
        modules.append(LpNormalize(2.0))
    return nn.Sequential(*modules)


class TransformerPooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.
    Source: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding.
    This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together (the order of the poolings is cls, max, mean, mean_sqrt when multiple
    poolings are specified

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Specify the pooling mode as one (or combination) of the below modes
        cls: Use the first token (CLS token) as text representations
        max: Use max in each dimension over all tokens.
        mean: Perform mean-pooling
        mean_sqrt: Perform mean-pooling, but divide by sqrt(input_length).
    """

    def __init__(
        self, word_embedding_dimension: int, pooling_mode: str, attention_mask_feat_name: str = "attention_mask"
    ):
        super(TransformerPooling, self).__init__()

        pooling_mode = pooling_mode.lower()
        self.pooling_mode_cls_token = False
        self.pooling_mode_max_tokens = False
        self.pooling_mode_mean_tokens = False
        self.pooling_mode_mean_sqrt_len_tokens = False
        for mode in pooling_mode.split(","):
            assert mode in [
                "mean",
                "max",
                "cls",
                "mean_sqrt",
            ], f"Invalid mode ({mode}) in pooling mode ({pooling_mode})"
            self.pooling_mode_cls_token |= mode == "cls"
            self.pooling_mode_max_tokens |= mode == "max"
            self.pooling_mode_mean_tokens |= mode == "mean"
            self.pooling_mode_mean_sqrt_len_tokens |= mode == "mean_sqrt"

        self.word_embedding_dimension = word_embedding_dimension

        pooling_mode_multiplier = sum(
            [
                self.pooling_mode_cls_token,
                self.pooling_mode_max_tokens,
                self.pooling_mode_mean_tokens,
                self.pooling_mode_mean_sqrt_len_tokens,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension
        self.attention_mask_feat_name = attention_mask_feat_name

    def extra_repr(self) -> str:
        return (
            f"word_embedding_dimension={self.word_embedding_dimension}, pooling_mode={self.get_pooling_mode_str()}"
            f", attention_mask_feat_name={self.attention_mask_feat_name}"
        )

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt")

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features[self.attention_mask_feat_name]

        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sequence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.pooling_output_dimension


class TextEmbedder(SageEmbedder):
    """
    Text embedding module that starts with a huggingface pretrained model, and embeds the input text
    that was tokenized using the corresponding pretrained huggingface tokenizer
    """

    def __init__(
        self,
        base_model_name: str,
        vocab_size: Optional[int] = None,
        pooling_mode: str = "cls",
        output_dim: int = 256,
        normalize: bool = True,
        input_id_feat_name: str = "input_ids",
        attention_mask_feat_name: str = "attention_mask",
        freeze_lm: bool = False,
        precision: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        if self.transformer.config.model_type in ["t5", "mt5", "umt5"]:
            assert pooling_mode == "mean", "T5 models only support mean pooling"
            self.transformer = self.transformer.encoder
        if freeze_lm:
            # give the option to freeze the language model
            for param in self.transformer.parameters():
                param.requires_grad = False
        if vocab_size is not None and vocab_size != self.transformer.config.vocab_size:
            self.transformer.resize_token_embeddings(vocab_size)
        self.pooler = TransformerPooling(
            word_embedding_dimension=self.transformer.config.hidden_size,
            pooling_mode=pooling_mode,
            attention_mask_feat_name=attention_mask_feat_name,
        )
        self.output_layer = _generate_mlp(
            self.pooler.get_sentence_embedding_dimension(),
            output_dim,
            layernorm=True,
            normalize=normalize,
            precision=precision,
        )
        self.input_id_feat_name = input_id_feat_name
        self.attention_mask_feat_name = attention_mask_feat_name

    def extra_repr(self) -> str:
        return f"input_id_feat_name={self.input_id_feat_name}, attention_mask_feat_name={self.attention_mask_feat_name}"

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_embeddings = self.transformer(
            feats[self.input_id_feat_name], attention_mask=feats.get(self.attention_mask_feat_name)
        )
        embs = self.pooler({"token_embeddings": token_embeddings[0], **feats})
        return self.output_layer(embs["sequence_embedding"])


class HashEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_hashes: int,
        vocab_size: int,
        embedding_dim: int,
        num_embeds: int,
        hash_weights: bool,
        normalize: bool = False,
    ):
        """
        Implements a bag of hash embeddings with sum pooling as described in https://arxiv.org/abs/1709.03933
        using num_hashes hash functions, we probe into `embeddings`, and then average the resulting embeddings
        using weights determined by `weight_emb`. if hash_weights is true, we assume inputs in the range [0, inf),
        hash the id used to lookup into weight_emb. otherwise, we all input ids must be less than vocab_size
        """
        super().__init__()
        self.embeddings = nn.EmbeddingBag(num_embeddings=num_embeds, embedding_dim=embedding_dim, mode="sum")
        self.weight_emb = nn.Embedding(vocab_size, num_hashes)
        self.num_hashes: Final[int] = num_hashes
        self.normalize: Final[bool] = normalize
        self.norm1 = nn.LayerNorm(embedding_dim)
        nn.init.xavier_normal_(self.embeddings.weight)
        nn.init.normal_(self.weight_emb.weight, 0.0, 1)

        def is_prime(x):
            for i in range(2, int(math.sqrt(x))):
                if x % i == 0:
                    return False
            return True

        def next_prime(n):
            while not is_prime(n):
                n += 1
            return n

        random.seed(1924031)

        def draw_hash(N: int) -> Tuple[int, int, int, int]:
            p = next_prime(random.randint(vocab_size, int(2**32)))
            a, b = random.randint(1, p), random.randint(1, p)
            return a, b, p, N

        self.hashes: Final[List[Tuple[int, int, int, int]]] = [draw_hash(num_embeds) for _ in range(num_hashes)]
        if hash_weights:
            self.weight_hash: Optional[Tuple[int, int, int, int]] = draw_hash(vocab_size)
        else:
            self.weight_hash: Optional[Tuple[int, int, int, int]] = None

    def _call_hash(self, a: int, b: int, p: int, N: int, x: torch.Tensor) -> torch.Tensor:
        return ((a * x + b) % p) % N

    def forward(self, input_ids, offsets: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None):
        if offsets is None:
            offsets = torch.arange(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        new_ids = torch.stack([self._call_hash(*hash_, input_ids) for hash_ in self.hashes], dim=1).view(-1)

        weight_ids = input_ids if self.weight_hash is None else self._call_hash(*self.weight_hash, input_ids)

        if weights is None:
            weights = self.weight_emb(weight_ids).view(-1)
        else:
            weights = (weights.unsqueeze(1) * self.weight_emb(weight_ids)).view(-1)
        embs = self.embeddings(new_ids, offsets * self.num_hashes, weights)
        embs = self.norm1(embs)

        if self.normalize:
            embs = nn.functional.normalize(embs, p=2.0, dim=1)
        return embs
