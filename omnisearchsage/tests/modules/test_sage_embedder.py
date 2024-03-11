import unittest

import numpy as np
import torch
from omnisearchsage.modules.sage_embedder import FeatureEmbedder
from omnisearchsage.modules.sage_embedder import VisualFeatureEmbedder


class FeatureEmbedderTestCase(unittest.TestCase):
    def test_basic(self):
        embedder = FeatureEmbedder(key="a")
        embs = torch.arange(15).view(5, 3)
        feats = {"a": embs}
        np.testing.assert_array_almost_equal(embs, embedder(feats))


class VisualFeatureEmbedderTestCase(unittest.TestCase):
    def test_basic(self):
        embedder = VisualFeatureEmbedder(key="a")

        embs = torch.tensor([[127, 0], [101, 254]], dtype=torch.uint8)
        feats = {"a": embs}

        expected_embs = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(expected_embs, embedder(feats))

    def test_structure(self):
        embedder_wo_unused_parameter = VisualFeatureEmbedder(key="a", use_unused_parameter=False)
        total_model_parameters_wo_unused_parameter = sum(p.numel() for p in embedder_wo_unused_parameter.parameters())
        assert total_model_parameters_wo_unused_parameter == 0

        embedder_with_unused_parameter = VisualFeatureEmbedder(key="a", use_unused_parameter=True)
        total_model_parameters_with_unused_parameter = sum(
            p.numel() for p in embedder_with_unused_parameter.parameters()
        )
        assert total_model_parameters_with_unused_parameter == 1


if __name__ == "__main__":
    unittest.main()
