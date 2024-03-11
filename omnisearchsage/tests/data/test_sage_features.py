import unittest
from unittest import mock

import pyarrow as pa
import torch
from omnisearchsage.common.types import EntityType
from omnisearchsage.data.sage_features import SageFeaturizerV2
from omnisearchsage.data.sage_features import check_tensor_feats


class CheckTensorFeatsTestCase(unittest.TestCase):
    def test_basic(self):
        torch.manual_seed(42)
        tensor_feats = {
            "a": torch.rand(3, dtype=torch.float),
            "b": torch.rand(3, dtype=torch.half),
            "c": torch.arange(3, dtype=torch.int32),
            "d": torch.arange(3, dtype=torch.int64),
        }

        check_tensor_feats(tensor_feats)

        tensor_feats["e"] = ["hello", "world"]
        with self.assertRaises(AssertionError):
            check_tensor_feats(tensor_feats)


class TestSageFeaturizerV2(unittest.TestCase):
    def setUp(self):
        self.featurizer = SageFeaturizerV2()
        self.featurizer.extract_batch = mock.MagicMock()

    def test_single_collate_fn(self):
        # create a pa.table with these values

        index = pa.Table.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([b"4", b"5"], type=pa.binary()),
                pa.array([b"7", b"8"], type=pa.binary()),
            ],
            names=["key", "entity_type", "feat", "meta"],
        )

        self.featurizer.single_collate_fn(index)
        self.featurizer.extract_batch.assert_called_with(
            {EntityType.SIGNATURE: pa.array([b"4", b"5"], type=pa.binary())}, {EntityType.SIGNATURE: ["1", "2"]}, None
        )

    def test_pair_collate_fn(self):
        pairs = pa.Table.from_arrays(
            [
                pa.array(["0", "1", "2"], type=pa.string()),
                pa.array(["00", "11", "22"], type=pa.string()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([b"3", b"4", b"5"], type=pa.binary()),
                pa.array([b"6", b"7", b"8"], type=pa.binary()),
                pa.array([b"33", b"44", b"55"], type=pa.binary()),
            ],
            names=["query_key", "cand_key", "query_entity_type", "cand_entity_type", "feat1", "feat2", "meta"],
        )

        self.featurizer.pair_collate_fn(pairs)
        self.featurizer.extract_batch.assert_called_with(
            {EntityType.SIGNATURE: pa.array([b"3", b"4", b"5", b"6", b"7", b"8"], type=pa.binary())},
            {EntityType.SIGNATURE: ["0", "1", "2", "00", "11", "22"]},
            [b"33", b"44", b"55"],
        )

    def test_triplet_collate_fn(self):
        pairs = pa.Table.from_arrays(
            [
                pa.array(["0", "1"], type=pa.string()),
                pa.array(["00", "11"], type=pa.string()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([b"3", b"4"], type=pa.binary()),
                pa.array([b"33", b"44"], type=pa.binary()),
                pa.array([b"6", b"7"], type=pa.binary()),
            ],
            names=["query_key", "cand_key", "query_entity_type", "cand_entity_type", "feat1", "feat2", "meta"],
        )
        index = pa.Table.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([EntityType.SIGNATURE, EntityType.SIGNATURE], type=pa.int32()),
                pa.array([b"4", b"5"], type=pa.binary()),
                pa.array([b"7", b"8"], type=pa.binary()),
            ],
            names=["key", "entity_type", "feat", "meta"],
        )

        self.featurizer.triplet_collate_fn(pairs, index)
        self.featurizer.extract_batch.assert_called_with(
            {EntityType.SIGNATURE: pa.array([b"3", b"4", b"33", b"44", b"4", b"5"], type=pa.binary())},
            {EntityType.SIGNATURE: ["0", "1", "00", "11", "1", "2"]},
            [b"6", b"7"],
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
