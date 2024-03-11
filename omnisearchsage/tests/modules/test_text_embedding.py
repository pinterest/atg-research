from __future__ import annotations

import re
import unittest
from unittest import mock

import numpy as np
import torch
from omnisearchsage.modules.text_embedding import HashEmbeddingBag
from omnisearchsage.modules.text_embedding import TextEmbedder
from omnisearchsage.modules.text_embedding import TransformerPooling
from torch import nn


class TestTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 10), nn.Linear(10, 20), nn.Linear(20, 32))

    def forward(self, x, attention_mask):
        return (self.mlp(x.float())[:, None],)  # noqa: T801


class TransformerPoolingTest(unittest.TestCase):
    def test_config(self) -> None:
        for mode in ["cls", "max", "mean", "mean_sqrt", "cls,mean", "max,mean,mean_sqrt"]:
            with self.subTest(mode=mode):
                pooler = TransformerPooling(word_embedding_dimension=64, pooling_mode=mode)
                self.assertEqual(pooler.get_sentence_embedding_dimension(), 64 * len(mode.split(",")))
                self.assertEqual(pooler.get_pooling_mode_str(), mode.replace(",", "+"))

        with self.subTest("Test exception for invalid mode"):
            with self.assertRaisesRegex(AssertionError, re.escape("Invalid mode (min) in pooling mode (mean,min)")):
                TransformerPooling(word_embedding_dimension=128, pooling_mode="mean,min")

    def test_pooling_modes(self) -> None:
        test_input = {
            "token_embeddings": torch.tensor([[[1, 0], [0, 1], [2, 3]]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
        mode_to_output = {
            "cls": torch.tensor([[1, 0]]),
            "mean": torch.tensor([[0.5, 0.5]]),
            "mean_sqrt": torch.tensor([[0.7071, 0.7071]]),
            "max": torch.tensor([[1, 1]]),
            "cls,mean": torch.tensor([[1, 0, 0.5, 0.5]]),
            "max,mean,mean_sqrt": torch.tensor([[1, 1, 0.5, 0.5, 0.7071, 0.7071]]),
        }

        for mode, expected_output in mode_to_output.items():
            with self.subTest(mode=mode):
                pooler = TransformerPooling(word_embedding_dimension=64, pooling_mode=mode)
                torch.testing.assert_close(pooler(test_input)["sequence_embedding"], expected_output)


class TextEmbedderTest(unittest.TestCase):
    @mock.patch("omnisearchsage.modules.text_embedding.AutoModel")
    def test_transformer_forward(self, mock_auto_model) -> None:
        torch.manual_seed(123321)
        test_model = TestTransformer()
        test_model.config = mock.MagicMock()
        test_model.config.hidden_size = 32
        mock_auto_model.from_pretrained.return_value = test_model

        m = TextEmbedder(output_dim=2, base_model_name="test_model_name")
        with torch.no_grad():
            orig_out = m({"input_ids": torch.tensor([[1, 1]]), "attention_mask": torch.tensor([[1, 1]])})

        mock_auto_model.from_pretrained.assert_called_with("test_model_name")

        np.testing.assert_array_almost_equal(orig_out, torch.tensor([[0.874492, 0.48504]]), decimal=5)


class HashEmbeddingBagTest(unittest.TestCase):
    def test_forward_no_hash_weights(self):
        torch.manual_seed(0)
        bag = HashEmbeddingBag(
            num_hashes=2,
            vocab_size=5,
            num_embeds=2,
            hash_weights=False,
            embedding_dim=7,
        )

        torch.manual_seed(0)
        lengths = torch.randint(0, 10, size=(20,))
        offsets = lengths.cumsum(0) - lengths[0]
        input_ids = torch.randint(0, 5, size=(lengths.sum().item(),))

        for scripting in [False, True]:
            with self.subTest(scripting=scripting):
                mod_bag = torch.jit.script(bag) if scripting else bag
                embs = mod_bag(input_ids, offsets)

                self.assertEqual((20, 7), embs.shape)
                self.assertAlmostEqual(114.081, embs.abs().sum().item(), places=2)


if __name__ == "__main__":
    unittest.main()
