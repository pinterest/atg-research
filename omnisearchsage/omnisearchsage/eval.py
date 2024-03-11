from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import os
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from omnisearchsage.common.eval_utils.containers import EvalEmbeddings
from omnisearchsage.common.eval_utils.containers import EvaluationMetadata
from omnisearchsage.common.eval_utils.utils import RECALL_10
from omnisearchsage.common.eval_utils.utils import EvaluationTask
from omnisearchsage.common.eval_utils.utils import draw_qp_qneg_dist_histogram
from omnisearchsage.common.eval_utils.utils import write_scalar_metrics
from omnisearchsage.common.types import EntityType
from omnisearchsage.common.utils.fs_utils import mkdir_p
from omnisearchsage.model import TowerState
from yaml import Dumper

if TYPE_CHECKING:
    from omnisearchsage.common.eval_utils.containers import EvaluationLoaderGroup
    from omnisearchsage.data.sage_features import SageBatch
    from omnisearchsage.model import OmniSearchSAGE
    from torch.utils.tensorboard import SummaryWriter

METRICS = [RECALL_10]

# default metric by eval prefix. Defaults to recall@10 if not specified
PREFIX_TO_METRIC = {"organic/repin": RECALL_10}


def _sanitize_labels(strings: Sequence[Union[str, bytes]]) -> List[str]:
    """
    Converts a sequence of strings that may contain bytes to only strings
    """
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in strings]


def extract_query_positive_pairs(
    model: OmniSearchSAGE,
    pair_loader: Iterable[SageBatch],
    print_freq: int = 50,
) -> Dict[TowerState, EvaluationTask]:
    """
    Extract query/positive pairs from a `StreamingDataLoader`

    Args:
        model: model to embed queries
        pair_loader: data loader with the pairs
        print_freq: print progress every `print_freq` batches

    Returns:
        EvaluationTask containing queries, positive sigs, query embeddings, positive embeddings, and extracted
            weights
    """
    model_mode = model.training
    model.train(False)

    query_ids, pos_ids = [], []
    pos_embs = defaultdict(list)
    query_embs = defaultdict(list)
    with torch.inference_mode():
        feats: SageBatch
        for i, feats in enumerate(pair_loader):
            if not feats.entity_types:
                continue
            embeddings = model.compute_embeddings(feats)
            query_entity_type = feats.query_entity_type
            cand_entity_type = feats.candidate_entity_type
            if query_entity_type == EntityType.SEARCH_QUERY:
                query_embs_data = model.extract_query_embs(feats=feats, embeddings=embeddings)
                query_embs[TowerState.LEARNED].append(query_embs_data["query_emb"].cpu().numpy())

                for cand_entity_type, tower_state in embeddings:
                    cand_emb = embeddings[cand_entity_type, tower_state]
                    pos_embs[tower_state].append(cand_emb.cpu().numpy())
                query_ids.extend(feats.search_queries[: query_embs_data["batch_size"]])
                pos_ids.extend(feats.keys.get(cand_entity_type)[query_embs_data["cand_start_index"] :])
            else:
                raise ValueError(f"Unknown query entity type {query_entity_type}")
            if i % print_freq == 0:
                print(f"Processed {len(query_ids)} pairs")
    print(f"Completed processing {len(query_ids)} pairs")
    tasks = {}
    if query_entity_type == EntityType.SEARCH_QUERY:
        queries = EvalEmbeddings(
            keys=_sanitize_labels(query_ids),
            entity_type=EntityType.SEARCH_QUERY,
            embeddings=np.vstack(query_embs[TowerState.LEARNED]),
        )

        common_properties = dict(
            keys=_sanitize_labels(pos_ids),
            entity_type=cand_entity_type,
        )
        for tower_state, embs in pos_embs.items():
            if len(embs) == 0:
                continue
            positives = EvalEmbeddings(embeddings=np.vstack(embs), **common_properties)
            tasks[tower_state] = EvaluationTask(queries=queries, positives=positives)
    else:
        raise ValueError(f"Unknown query entity type {query_entity_type}")
    model.train(mode=model_mode)
    return tasks


def extract_index(
    model: OmniSearchSAGE,
    index_loader: Iterable[SageBatch],
    print_freq: int = 50,
) -> Dict[TowerState, EvaluationTask]:
    """
    Extract index embeddings from a `SearchSageIndexIterator`. Currently the model is purely for distillation,
    so there is no dependency on the model itself
    Args:
        model: Model to evaluate
        index_loader: `SearchSageIndexIterator` to read from
        print_freq: print progress every `print_freq` batches

    Returns:
        EvaluationTask containing image signatures, embeddings
    """
    model_mode = model.training
    model.train(False)

    all_sigs = []
    all_embs = defaultdict(list)

    with torch.inference_mode():
        for i, feats in enumerate(index_loader):
            cand_entity_type = feats.candidate_entity_type
            embeddings = model.compute_embeddings(feats)
            for cand_entity_type, tower_state in embeddings:
                emb = embeddings[cand_entity_type, tower_state]
                all_embs[tower_state].append(emb.cpu().numpy())
            all_sigs.extend(feats.keys.get(cand_entity_type))
            if i % print_freq == 0:
                print(f"Processed {len(all_sigs)} index sigs")

    tasks = {}
    common_properties = dict(keys=_sanitize_labels(all_sigs), entity_type=cand_entity_type)
    for tower_state, embs in all_embs.items():
        if len(embs) == 0:
            continue
        negatives = EvalEmbeddings(embeddings=np.vstack(embs).astype(np.float16), **common_properties)
        tasks[tower_state] = EvaluationTask(negatives=negatives)
    model.train(mode=model_mode)
    return tasks


def _do_eval(metadata: EvaluationMetadata, eval_task: EvaluationTask) -> Dict[str, Dict[str, float]]:
    """
    Runs the complete eval for a given EvaluationMetadata and EvaluationTask
    """
    metric_values = {metric: float(metric.compute(eval_task.positive_ranks)) for metric in METRICS}
    summary_writer = metadata.summary_writer

    dist_hist_fig = draw_qp_qneg_dist_histogram(eval_task, num_negative_rows=500)
    if summary_writer is not None:
        summary_writer.add_scalars(
            metadata.prefix,
            {k.metric_name: v for k, v in metric_values.items()},
            global_step=metadata.iteration,
        )
        for m in METRICS:
            tag = os.path.join(metadata.prefix, m.metric_name)
            summary_writer.add_text(tag, m.text_format_string.format(metric_values[m]), metadata.iteration)

        summary_writer.add_figure(
            tag=os.path.join(metadata.prefix, "distance_distribution"),
            figure=dist_hist_fig,
            global_step=metadata.iteration,
        )

    dist_hist_path = os.path.join(metadata.absolute_prefix, "dist_hist.svg")
    dist_hist_fig.savefig(dist_hist_path, bbox_inches="tight")

    if eval_task.queries.entity_type == EntityType.SEARCH_QUERY:
        yaml_out = write_scalar_metrics(metadata, eval_task, METRICS)
    else:
        yaml_out = write_scalar_metrics(metadata, eval_task, METRICS)
    return yaml_out


def evaluate_pair_task(
    *,
    query_name: str,
    index_name: str,
    qp_embs: EvaluationTask,
    index_embs: EvaluationTask,
    results_dict: Dict[str, Any],
    subdims: Sequence[int],
    iteration: int,
    run_dir: str,
    summary_writer: Optional[SummaryWriter] = None,
):
    def normalize(x: np.ndarray) -> np.ndarray:
        eps: float = 1e-12
        denom = np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)
        denom = np.broadcast_to(denom, x.shape)
        return x / denom

    for dim in subdims:
        if dim > qp_embs.positives.embeddings.shape[1]:
            continue

        query_name_dim = f"dim_{dim}/{query_name}" if dim != qp_embs.queries.embeddings.shape[1] else query_name
        prefix = f"{index_name}/{query_name_dim}"

        metadata = EvaluationMetadata(
            summary_writer=summary_writer,
            prefix=prefix,
            run_dir=run_dir,
            iteration=iteration,
            metric_to_plot=PREFIX_TO_METRIC.get(prefix, RECALL_10),
        )

        eval_task = EvaluationTask(
            queries=EvalEmbeddings(
                keys=qp_embs.queries.keys,
                entity_type=qp_embs.queries.entity_type,
                embeddings=normalize(qp_embs.queries.embeddings[:, :dim])
                if dim != qp_embs.queries.embeddings.shape[1]
                else qp_embs.queries.embeddings,
            ),
            positives=EvalEmbeddings(
                keys=qp_embs.positives.keys,
                entity_type=qp_embs.positives.entity_type,
                embeddings=(
                    normalize(qp_embs.positives.embeddings[:, :dim])
                    if dim != qp_embs.queries.embeddings.shape[1]
                    else qp_embs.positives.embeddings
                ),
            ),
            negatives=EvalEmbeddings(
                keys=index_embs.negatives.keys,
                entity_type=index_embs.negatives.entity_type,
                embeddings=(
                    normalize(index_embs.negatives.embeddings[:, :dim])
                    if dim != qp_embs.queries.embeddings.shape[1]
                    else index_embs.negatives.embeddings
                ),
            ),
        )

        results_dict[index_name][query_name_dim] = _do_eval(metadata, eval_task)


def create_evaluation(
    model, eval_groups: List[EvaluationLoaderGroup], subdims: Sequence[int] = (16, 32, 64, 128, 256)
) -> Callable[[int, str, Optional[SummaryWriter], bool], None]:
    """
    Generates an evaluation function that will process all of eval_groups.
    This is awfully similar to the graphsage function. Probably could be worth
    unifying these two functions at some point if we can come up with nice abstractions
    """

    def evaluate_embedding_model(
        iteration: int,
        run_dir: str,
        summary_writer: Optional[SummaryWriter] = None,
        run_full_eval: bool = False,
    ) -> None:
        """
        Function to call for evaluation. This runs _to_eval for each group within eval_groups
        and also saves the extracted embeddings and signatures to the eval directory so they can easily be
        analyzed offline.

        At a high level, for each unique (index_name, pairs_name) in eval_groups
        given some index of signatures, and some (query, engaged/relevant signature) pairs,
        we rank the engaged/relevant sig among the index of signatures, and then using those ranks of positive
        examples, compute each metric in METRICS using those ranks (for example R@1, R@10)

        Args:
            iteration: current iteration
            run_dir: base directory for eval
            summary_writer: tensorboardx summary writer
            run_full_eval: whether or not to run the full evaluation. the main difference is whether or not
                this will create a visualization
        """
        mkdir_p(run_dir)
        eval_start = time.time()
        results_dict = {}
        for group in eval_groups:
            print(f"Extraction index features for {group.index_name}")
            index_embs = extract_index(model, group.index_loader)
            for tower_state in index_embs:
                results_dict[group.index_name + "_" + tower_state.name.lower()] = {}
            for (
                query_name,
                query_positive_loader,
            ) in group.query_positive_loaders.items():
                print(f"index={group.index_name}\teval_set={query_name}")

                qp_embs = extract_query_positive_pairs(model, query_positive_loader)
                for tower_state in qp_embs:
                    print(
                        f"index={group.index_name}\teval_set={query_name}\ttower_state={tower_state.name}: Calculating"
                    )
                    evaluate_pair_task(
                        query_name=query_name,
                        index_name=group.index_name + "_" + tower_state.name.lower(),
                        qp_embs=qp_embs[tower_state],
                        index_embs=index_embs[tower_state],
                        results_dict=results_dict,
                        subdims=subdims,
                        iteration=iteration,
                        summary_writer=summary_writer,
                        run_dir=run_dir,
                    )

        results_dict["iteration"] = iteration
        yaml_str = yaml.dump(results_dict, Dumper=Dumper, default_flow_style=False, sort_keys=False)
        print(yaml_str)

        with open(os.path.join(run_dir, "results.yaml"), "w") as f:
            f.write(yaml_str)

        eval_end = time.time()
        print(f"Evaluation at iteration {iteration} complete in {eval_end - eval_start:.2f} seconds")
        if summary_writer is not None:
            summary_writer.add_scalar(
                "eval_timings/eval_duration",
                eval_end - eval_start,
                global_step=iteration,
            )

    return evaluate_embedding_model
