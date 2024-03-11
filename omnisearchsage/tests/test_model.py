from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

import unittest
from unittest import mock

import pytest
import torch
import torch.nn.functional
from omnisearchsage.common.types import EntityType
from omnisearchsage.data.sage_features import SageBatch
from omnisearchsage.data.sage_features import TaskName
from omnisearchsage.feature_consts import STRING_FEATURES
from omnisearchsage.model import OmniSearchSAGE
from omnisearchsage.model import OmniSearchSAGEQueryEmbedder
from omnisearchsage.model import OmniSearchSAGETrainTask
from omnisearchsage.model import TowerState
from omnisearchsage.modules.text_embedding import TextEmbedder
from omnisearchsage.modules.tokenization import MultiVocabTokenizer
from omnisearchsage.train import create_model
from torch import nn
from transformers import T5Config

if TYPE_CHECKING:
    from torch import Tensor


class TestTokenizer(nn.Module):
    def __init__(self, input: Dict[str, Tensor]) -> None:
        super().__init__()
        self.input = input
        self.texts = ["a"]

    def forward(self, texts: List[str], normalize: bool = False) -> Dict[str, Tensor]:
        self.texts = texts
        return self.input

    def encode_batch(self, texts: List[str], normalize: bool = False):
        return self(texts, normalize)


class TestTransformer(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.ones(5, 4))
        self.config = T5Config(hidden_size=hidden_size, vocab_size=256300)
        self.output = torch.rand(4, 7, 768)

    def forward(self, text_ids, **kwargs):
        return self.output.to(self.unused.device), None

    @property
    def encoder(self) -> nn.Module:
        return self


@pytest.mark.gpu
@unittest.skipIf(not torch.cuda.is_available(), "gpu only test")
class SearchSAGEQueryEmbedderTest(unittest.TestCase):
    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def setUp(self, auto_model) -> None:
        torch.manual_seed(0)
        auto_model.from_pretrained.return_value = TestTransformer()
        base_model_name = "test_model"

        encoded_ids = (torch.arange(28) + 500).view(4, -1)
        mask = torch.rand(4, 7) > 0.1
        input = {"input_ids": encoded_ids, "attention_mask": mask}
        self.device = torch.device("cuda")
        self.model = OmniSearchSAGEQueryEmbedder(
            embedder=TextEmbedder(
                base_model_name,
                vocab_size=256300,
                precision=torch.float16,
                pooling_mode="mean",
                input_id_feat_name="test_text_input_ids",
                attention_mask_feat_name="test_text_attention_mask",
            ),
            tokenizer=TestTokenizer(input),
            feature_name="test_text",
            device=self.device,
        ).to(self.device)
        self.input = {f"test_text_{k}": v for k, v in input.items()}

    def test_forward(self) -> None:
        model = self.model
        self.assertFalse(model.should_tokenize)

        def assert_embs(embs):
            torch.testing.assert_close(
                torch.norm(embs, dim=1), torch.ones(embs.shape[0], dtype=torch.float16, device=self.device)
            )
            torch.testing.assert_close((embs**2).sum(), torch.tensor(4.0, dtype=torch.float16, device=self.device))
            torch.testing.assert_close(embs.sum(), torch.tensor(3.1836, dtype=torch.float16, device=self.device))

        assert_embs(model(self.input).detach())
        model.toggle_tokenization(tokenize=True)
        self.assertTrue(model.should_tokenize)

        assert_embs(model({}, {"test_text": ["a"]}).detach())


class TestVocabTokenizer(nn.Module):
    def __len__(self) -> int:
        return 4


@pytest.mark.gpu
@unittest.skipIf(not torch.cuda.is_available(), "gpu only test")
class OmniSearchSAGETest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(MultiVocabTokenizer, "default")
        t = patcher.start()
        t.return_value = TestVocabTokenizer()
        self.addCleanup(patcher.stop)
        self.query_base_model_name = "distilbert-base-multilingual-cased"

    def transformer_side_effect(self, input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor]:
        torch.manual_seed(0)
        return (torch.rand(input_ids.shape[0], input_ids.shape[1], 768, device=torch.device("cuda")),)

    def test_encode_decode_embedder_key(self) -> None:
        entity_type = EntityType.SIGNATURE
        tower_state = TowerState.FIXED_GS
        self.assertEqual(OmniSearchSAGE.encode_embedder_key(entity_type, tower_state), "SIGNATURE::FIXED_GS")
        self.assertTupleEqual(OmniSearchSAGE.decode_embedder_key("SIGNATURE::FIXED_GS"), (entity_type, tower_state))
        self.assertTupleEqual(
            OmniSearchSAGE.decode_embedder_key(OmniSearchSAGE.encode_embedder_key(entity_type, tower_state)),
            (entity_type, tower_state),
        )

    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def test_forward(self, auto_model: mock.MagicMock) -> None:
        device = torch.device("cuda")
        torch.manual_seed(0)
        transformer = auto_model.from_pretrained.return_value
        transformer.config.hidden_size = 768
        transformer.side_effect = self.transformer_side_effect

        model = create_model(query_base_model_name=self.query_base_model_name, device=device)
        gs_embs = torch.nn.functional.normalize(torch.randn(4, 256))
        ue_v4_embs = torch.randint(0, 255, (4, 128), dtype=torch.uint8)
        item_embs = torch.nn.functional.normalize(torch.randn(4, 256))
        encoded_ids = (torch.arange(28) + 500).view(4, -1)
        mask = torch.rand(4, 7) > 0.1
        queries = [f"q_{i}" for i in range(12)]

        batches = [
            SageBatch(
                keys={
                    EntityType.SIGNATURE: [f"s_{i}" for i in range(8)],
                    EntityType.SEARCH_QUERY: queries[:4],
                },
                tensor_feats={
                    EntityType.SIGNATURE: {
                        "gs_v5": gs_embs.repeat(2, 1),
                        "ue_v4": ue_v4_embs.repeat(2, 1),
                        "item_is_v2": item_embs.repeat(2, 1),
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize([f"s_{i}" for i in range(8)]),
                        **{
                            f"{feature_name}_input_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                        **{
                            f"{feature_name}_offsets": torch.tensor([0, 2, 2, 3, 4, 6, 6, 7])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                    },
                    EntityType.SEARCH_QUERY: {
                        "query_text_input_ids": encoded_ids,
                        "query_text_attention_mask": mask,
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize(queries[:4]),
                    },
                },
                task_name=TaskName.METRIC_LEARNING,
                candidate_entity_type=EntityType.SIGNATURE,
                query_entity_type=EntityType.SEARCH_QUERY,
                num_pairs=4,
            ),
            SageBatch(
                keys={
                    EntityType.ITEM: [f"s_{i}" for i in range(8)],
                    EntityType.SEARCH_QUERY: queries[:4],
                },
                tensor_feats={
                    EntityType.ITEM: {
                        "gs_v5_feat": gs_embs.repeat(8, 5, 1),
                        "gs_v5_mask": torch.ones(8, 20, 1, dtype=torch.bool),
                        "item_is_v2": gs_embs.repeat(2, 1),
                        "ue_v4_feat": ue_v4_embs.repeat(8, 5, 1),
                        "ue_v4_mask": torch.ones(8, 20, 1, dtype=torch.bool),
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize([f"item_{i}" for i in range(8)]),
                        **{
                            f"{feature_name}_input_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
                            for feature_name in STRING_FEATURES[EntityType.ITEM]
                        },
                        **{
                            f"{feature_name}_offsets": torch.tensor([0, 2, 2, 3, 4, 6, 6, 7])
                            for feature_name in STRING_FEATURES[EntityType.ITEM]
                        },
                    },
                    EntityType.SEARCH_QUERY: {
                        "query_text_input_ids": encoded_ids,
                        "query_text_attention_mask": mask,
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize(queries[:4]),
                    },
                },
                task_name=TaskName.METRIC_LEARNING,
                candidate_entity_type=EntityType.ITEM,
                query_entity_type=EntityType.SEARCH_QUERY,
                num_pairs=4,
            ),
            SageBatch(
                keys={
                    EntityType.SIGNATURE: [f"s_{i}" for i in range(8)],
                    EntityType.SEARCH_QUERY: queries[:4],
                },
                tensor_feats={
                    EntityType.SIGNATURE: {
                        "gs_v5": gs_embs.repeat(2, 1),
                        "ue_v4": ue_v4_embs.repeat(2, 1),
                        "item_is_v2": item_embs.repeat(2, 1),
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize([f"s_{i}" for i in range(8)]),
                        **{
                            f"{feature_name}_input_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                        **{
                            f"{feature_name}_offsets": torch.tensor([0, 2, 2, 3, 4, 6, 6, 7])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                    },
                    EntityType.SEARCH_QUERY: {
                        "query_text_input_ids": encoded_ids,
                        "query_text_attention_mask": mask,
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize(queries[:4]),
                    },
                },
                task_name=TaskName.METRIC_LEARNING,
                candidate_entity_type=EntityType.SIGNATURE,
                query_entity_type=EntityType.SEARCH_QUERY,
                num_pairs=4,
            ),
            SageBatch(
                keys={
                    EntityType.SEARCH_QUERY: queries,
                },
                tensor_feats={
                    EntityType.SEARCH_QUERY: {
                        "query_text_input_ids": (torch.arange(84) + 500).view(12, -1),
                        "query_text_attention_mask": torch.rand(12, 7) > 0.1,
                        "id_hash": torch.ops.pinterest_ops.hash_tokenize(queries),
                    },
                },
                task_name=TaskName.METRIC_LEARNING,
                candidate_entity_type=EntityType.SEARCH_QUERY,
                query_entity_type=EntityType.SEARCH_QUERY,
                num_pairs=4,
            ),
        ]

        losses = model(batches)

        expected_losses = {
            "search_query_signature_learned_16_in_batch": 0.2445068359375,
            "search_query_signature_learned_16_rnd_neg": 0.2626953125,
            "search_query_signature_learned_32_in_batch": 0.202880859375,
            "search_query_signature_learned_32_rnd_neg": 0.2213134765625,
            "search_query_signature_learned_64_in_batch": 0.1939697265625,
            "search_query_signature_learned_64_rnd_neg": 0.212158203125,
            "search_query_signature_learned_128_in_batch": 0.1373291015625,
            "search_query_signature_learned_128_rnd_neg": 0.1544189453125,
            "search_query_signature_learned_256_in_batch": 1.95703125,
            "search_query_signature_learned_256_rnd_neg": 2.232421875,
            "search_query_signature_learned_512_in_batch": 1.748046875,
            "search_query_signature_learned_512_rnd_neg": 1.9619140625,
            "search_query_signature_fixed_gs_16_in_batch": 0.364013671875,
            "search_query_signature_fixed_gs_16_rnd_neg": 0.379638671875,
            "search_query_signature_fixed_gs_32_in_batch": 0.226806640625,
            "search_query_signature_fixed_gs_32_rnd_neg": 0.241943359375,
            "search_query_signature_fixed_gs_64_in_batch": 0.11444091796875,
            "search_query_signature_fixed_gs_64_rnd_neg": 0.1260986328125,
            "search_query_signature_fixed_gs_128_in_batch": 0.10870361328125,
            "search_query_signature_fixed_gs_128_rnd_neg": 0.12335205078125,
            "search_query_signature_fixed_gs_256_in_batch": 2.072265625,
            "search_query_signature_fixed_gs_256_rnd_neg": 2.25,
            "search_query_search_query_learned_16_in_batch": 0.0888671875,
            "search_query_search_query_learned_16_rnd_neg": 0.103515625,
            "search_query_search_query_learned_32_in_batch": 0.09197998046875,
            "search_query_search_query_learned_32_rnd_neg": 0.10614013671875,
            "search_query_search_query_learned_64_in_batch": 0.0860595703125,
            "search_query_search_query_learned_64_rnd_neg": 0.09942626953125,
            "search_query_search_query_learned_128_in_batch": 0.08428955078125,
            "search_query_search_query_learned_128_rnd_neg": 0.09918212890625,
            "search_query_search_query_learned_256_in_batch": 1.6904296875,
            "search_query_search_query_learned_256_rnd_neg": 2.025390625,
            "search_query_search_query_learned_512_in_batch": 1.6640625,
            "search_query_search_query_learned_512_rnd_neg": 2.00390625,
            "search_query_item_learned_16_in_batch": 0.2161865234375,
            "search_query_item_learned_16_rnd_neg": 0.224609375,
            "search_query_item_learned_32_in_batch": 0.16064453125,
            "search_query_item_learned_32_rnd_neg": 0.169189453125,
            "search_query_item_learned_64_in_batch": 0.1612548828125,
            "search_query_item_learned_64_rnd_neg": 0.169677734375,
            "search_query_item_learned_128_in_batch": 0.11456298828125,
            "search_query_item_learned_128_rnd_neg": 0.12384033203125,
            "search_query_item_learned_256_in_batch": 1.205078125,
            "search_query_item_learned_256_rnd_neg": 1.4267578125,
            "search_query_item_learned_512_in_batch": 0.89404296875,
            "search_query_item_learned_512_rnd_neg": 1.1318359375,
            "search_query_item_fixed_is_16_in_batch": 0.34326171875,
            "search_query_item_fixed_is_16_rnd_neg": 0.35302734375,
            "search_query_item_fixed_is_32_in_batch": 0.205322265625,
            "search_query_item_fixed_is_32_rnd_neg": 0.2158203125,
            "search_query_item_fixed_is_64_in_batch": 0.0882568359375,
            "search_query_item_fixed_is_64_rnd_neg": 0.10235595703125,
            "search_query_item_fixed_is_128_in_batch": 0.0859375,
            "search_query_item_fixed_is_128_rnd_neg": 0.0986328125,
            "search_query_item_fixed_is_256_in_batch": 1.5,
            "search_query_item_fixed_is_256_rnd_neg": 1.7646484375,
        }

        for k, v in expected_losses.items():
            with self.subTest(k=k):
                torch.testing.assert_close(
                    losses["loss_components"][k],
                    torch.tensor(v, dtype=torch.float16, device=device),
                    atol=1e-3,
                    rtol=1e-2,
                )
        total_loss = sum(expected_losses.values()) / 11
        torch.testing.assert_close(
            losses["total_loss"], torch.tensor(total_loss, dtype=torch.float16, device=device), atol=1e-3, rtol=1e-2
        )

    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def test_eval_forward_query(self, auto_model: mock.MagicMock) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        transformer = auto_model.from_pretrained.return_value
        transformer.config.hidden_size = 768
        transformer.side_effect = self.transformer_side_effect

        model = create_model(query_base_model_name=self.query_base_model_name, device=device)
        model.eval()
        encoded_ids = (torch.arange(28) + 500).view(4, -1)
        mask = torch.rand(4, 7) > 0.1

        with torch.no_grad():
            embs = model.compute_embeddings(
                SageBatch(
                    keys={
                        EntityType.SEARCH_QUERY: ["q"] * 4,
                    },
                    tensor_feats={
                        EntityType.SEARCH_QUERY: {
                            "query_text_input_ids": encoded_ids,
                            "query_text_attention_mask": mask,
                        },
                    },
                    task_name=TaskName.METRIC_LEARNING,
                )
            )
            preds = embs[EntityType.SEARCH_QUERY, TowerState.LEARNED]
            torch.testing.assert_close(
                torch.norm(preds, dim=1), torch.ones(preds.shape[0], device=device, dtype=torch.float16)
            )
            self.assertAlmostEqual((preds**2).sum().item(), 4.0, places=4)
            self.assertAlmostEqual(preds.sum().item(), 5.48046875, places=4)

    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def test_eval_forward_pin(self, auto_model: mock.MagicMock) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        transformer = auto_model.from_pretrained.return_value
        transformer.config.hidden_size = 768
        transformer.side_effect = self.transformer_side_effect

        model = create_model(query_base_model_name=self.query_base_model_name, device=device)
        model.eval()
        gs_embs = torch.nn.functional.normalize(torch.randn(4, 256))
        is_embs = torch.nn.functional.normalize(torch.randn(4, 256))
        ue_v4_embs = torch.randint(0, 255, (4, 128), dtype=torch.uint8)
        for dataset in [OmniSearchSAGETrainTask.ORGANIC, OmniSearchSAGETrainTask.NATIVE]:
            batch = SageBatch(
                keys={
                    EntityType.SIGNATURE: ["s"] * 4,
                },
                tensor_feats={
                    EntityType.SIGNATURE: {
                        "gs_v5": gs_embs,
                        "ue_v4": ue_v4_embs,
                        "item_is_v2": is_embs,
                        **{
                            f"{feature_name}_input_ids": torch.tensor([0, 1, 2, 3])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                        **{
                            f"{feature_name}_offsets": torch.tensor([0, 2, 2, 3])
                            for feature_name in STRING_FEATURES[EntityType.SIGNATURE]
                        },
                    },
                },
                task_name=TaskName.METRIC_LEARNING,
            )
            embs = model.compute_embeddings(batch)

            expected_keys = {
                (EntityType.SIGNATURE, TowerState.LEARNED),
                (EntityType.SIGNATURE, TowerState.FIXED_GS),
            }

            with self.subTest(dataset=dataset, test="keys"):
                self.assertSetEqual(set(embs.keys()), expected_keys)

            with self.subTest(dataset=dataset, emb="learned_embeddings"):
                with torch.no_grad():
                    preds = embs[EntityType.SIGNATURE, TowerState.LEARNED]
                    torch.testing.assert_close(
                        torch.norm(preds, dim=1), torch.ones(preds.shape[0], dtype=torch.float16, device=device)
                    )
                    torch.testing.assert_close(
                        (preds**2).sum(), torch.tensor(4.0, dtype=torch.float16, device=device)
                    )
                    torch.testing.assert_close(
                        preds.sum(), torch.tensor(1.17866265625, dtype=torch.float16, device=device)
                    )

            with self.subTest(dataset=dataset, emb="fixed_embeddings_gs"):
                with torch.no_grad():
                    preds = embs[EntityType.SIGNATURE, TowerState.FIXED_GS]
                    torch.testing.assert_close(preds, gs_embs.to(torch.float16).to(device))

    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def test_eval_forward_item(self, auto_model: mock.MagicMock) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        transformer = auto_model.from_pretrained.return_value
        transformer.config.hidden_size = 768
        transformer.side_effect = self.transformer_side_effect

        model = create_model(query_base_model_name=self.query_base_model_name, device=device)
        model.eval()
        gs_embs = torch.nn.functional.normalize(torch.randn(4, 20, 256))
        is_embs = torch.nn.functional.normalize(torch.randn(4, 256))
        ue_v4_embs = torch.randint(0, 255, (4, 20, 128), dtype=torch.uint8)
        dataset = OmniSearchSAGETrainTask.ITEM
        batch = SageBatch(
            keys={
                EntityType.ITEM: ["i"] * 4,
            },
            tensor_feats={
                EntityType.ITEM: {
                    "gs_v5_feat": gs_embs,
                    "item_is_v2": is_embs,
                    "ue_v4_feat": ue_v4_embs,
                    "gs_v5_mask": torch.ones(4, 20, 1, dtype=torch.bool),
                    "ue_v4_mask": torch.ones(4, 20, 1, dtype=torch.bool),
                    **{
                        f"{feature_name}_input_ids": torch.tensor([0, 1, 2, 3])
                        for feature_name in STRING_FEATURES[EntityType.ITEM]
                    },
                    **{
                        f"{feature_name}_offsets": torch.tensor([0, 2, 2, 3])
                        for feature_name in STRING_FEATURES[EntityType.ITEM]
                    },
                },
            },
            task_name=TaskName.METRIC_LEARNING,
        )
        embs = model.compute_embeddings(batch)

        expected_keys = {
            (EntityType.ITEM, TowerState.LEARNED),
            (EntityType.ITEM, TowerState.FIXED_IS),
        }

        with self.subTest(test="keys"):
            self.assertSetEqual(set(embs.keys()), expected_keys)

        with self.subTest(dataset=dataset, emb="learned_embeddings"):
            with torch.no_grad():
                preds = embs[EntityType.ITEM, TowerState.LEARNED]
                torch.testing.assert_close(
                    torch.norm(preds, dim=1), torch.ones(preds.shape[0], dtype=torch.float16, device=device)
                )
                torch.testing.assert_close((preds**2).sum(), torch.tensor(4.0, dtype=torch.float16, device=device))
                torch.testing.assert_close(
                    preds.sum(), torch.tensor(0.741251875, dtype=torch.float16, device=device), atol=1e-2, rtol=1e-2
                )

        with self.subTest(dataset=dataset, emb="fixed_embeddings_is"):
            with torch.no_grad():
                preds = embs[EntityType.ITEM, TowerState.FIXED_IS]
                torch.testing.assert_close(preds, is_embs.to(torch.float16).to(device))


if __name__ == "__main__":
    unittest.main()
