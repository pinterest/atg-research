from __future__ import annotations

import math
import unittest
from unittest import mock
from unittest.mock import call

import numpy as np
import torch
import torch.nn.functional as F
from omnisearchsage.modules.metric_learning_softmax import MultiDimSoftmaxCorrectionLossV2
from omnisearchsage.modules.metric_learning_softmax import SoftmaxCorrectionLoss
from omnisearchsage.modules.metric_learning_softmax import SoftmaxNegativeMode
from torch.utils.tensorboard import SummaryWriter


class SoftmaxCorrectionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query_feat = torch.tensor([[0.0, 1], [1, 0], [1, 1], [1, 1]])
        self.positive_feat = F.normalize(torch.tensor([[3.0, 3], [11, 11], [14, 14], [1, 0]]), p=2.0, dim=1)
        self.negative_feat = F.normalize(torch.tensor([[1.0, 2]]), p=2.0, dim=1)

    def test_repr(self):
        self.maxDiff = 1000
        self.assertEqual(
            str(SoftmaxCorrectionLoss(synchronize_pos=True, d=3, w=24)),
            (
                "SoftmaxCorrectionLoss(\n"
                "  neg_mode=MIXED_SUM, enable_spc=True, max_in_batch_negatives=None\n"
                "  (conditional_ct): ConditionalProbabilityEstimator(\n"
                "    (q_freq_est): CountMinSketch(w=24,d=3)\n"
                "    (qp_freq_est): CountMinSketch(w=24,d=3)\n"
                "  )\n"
                "  (in_batch_neg_ct): NegativeCounter(\n"
                "    (freq_est): CountMinSketch(w=24,d=3)\n"
                "  )\n"
                "  (rnd_neg_ct): NegativeCounter(\n"
                "    (freq_est): CountMinSketch(w=24,d=3)\n"
                "  )\n"
                ")"
            ),
        )

    def test_forward(self):
        query_ids = torch.tensor([1, 3, 7, 9])
        pos_ids = torch.tensor([3, 7, 1, 1])
        neg_ids = torch.tensor([4])
        similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, synchronize_pos=True)
        losses = similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=query_ids,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        with self.subTest("first call"):
            np.testing.assert_array_almost_equal(torch.stack(losses).detach(), [3.0478, 1.9896], decimal=3)

        # update cms table
        similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=torch.tensor([7, 7, 9, 7]),
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        losses = similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=query_ids,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        with self.subTest("third call"):
            # notice loss is not [3.0478, 1.9896] even with same input
            np.testing.assert_array_almost_equal(torch.stack(losses).detach(), [2.9703, 1.9619], decimal=3)

        with self.subTest("test write summary"):
            self._test_summary(similarity_loss)

    def _test_summary(self, similarity_loss: SoftmaxCorrectionLoss) -> None:
        summary_writer = mock.MagicMock(spec=SummaryWriter)

        similarity_loss.write_summary(summary_writer, epoch=0, iteration=85, total_iterations=100)

        summary_writer.assert_not_called()

        similarity_loss.write_summary(summary_writer, epoch=0, iteration=100, total_iterations=1000)

        summary_writer.add_scalar.assert_has_calls(
            [
                call("stats/num_posnegs", 4, global_step=100),
                call("stats/num_pairs", 4, global_step=100),
            ]
        )

        similarity_loss.write_summary(summary_writer, epoch=0, iteration=1000, total_iterations=200)

        summary_writer.add_scalar.assert_has_calls(
            [
                call("stats/num_posnegs", 4, global_step=1000),
                call("stats/num_pairs", 4, global_step=1000),
            ]
        )

        self.assertEqual(summary_writer.add_scalar.call_count, 6)
        summary_writer.add_scalar.assert_any_call(
            global_step=1000, scalar_value=mock.ANY, tag="debug_metrics/loss_temperature"
        )
        summary_writer.add_scalar.assert_called_with(global_step=1000, scalar_value=2, tag="debug_metrics/max_pos_freq")

        self.assertEqual(summary_writer.add_histogram.call_count, 3)
        for name in ["p_given_q_probs", "rnd_neg_probs", "in_batch_neg_probs"]:
            with self.subTest(name=name):
                summary_writer.add_histogram.assert_any_call(tag=f"stats/{name}", values=mock.ANY, global_step=1000)

    def test_forward_with_cap(self):
        query_ids = torch.tensor([1, 3, 7, 9, 10, 3, 15, 2])
        pos_ids = torch.tensor([3, 7, 1, 1, 8, 9, 11, 13])
        neg_ids = torch.tensor([4])
        similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, synchronize_pos=True, max_in_batch_negatives=2)
        with mock.patch.object(
            similarity_loss,
            "scale_correct_and_mask",
            wraps=similarity_loss.scale_correct_and_mask,
        ) as mock_scale:
            losses = similarity_loss(
                query=self.query_feat.repeat_interleave(2, dim=0),
                pos=self.positive_feat.repeat_interleave(2, dim=0),
                rnd_neg=self.negative_feat,
                query_ids=query_ids,
                pos_ids=pos_ids,
                neg_ids=neg_ids,
            )
        with self.subTest("first call"):
            self.assertFalse(torch.tensor(losses).isnan().any().item())
        with self.subTest("correct logit shapes"):
            self.assertEqual(2, mock_scale.call_count)
            (input_tensor,), kwargs = mock_scale.call_args_list[0]
            with self.subTest("input tensor"):
                self.assertEqual(3, input_tensor.size(1))
            with self.subTest("is_in_batch_neg"):
                self.assertTrue(kwargs["is_in_batch_neg"])

    def prepare_for_test_manual_correction(self):
        qp = (self.query_feat * self.positive_feat).sum(1, keepdim=True)
        qpn = self.query_feat @ self.positive_feat.t()
        qn = self.query_feat @ self.negative_feat.t()

        randn_logits = torch.cat((qp, qn), dim=1) / 0.4
        randn_logits[:, 0] -= torch.log(torch.tensor([0.5, 0.5, 1.0, 1.0]))
        randn_logits[:, 1:] -= torch.log(torch.tensor([1.0]))
        randn_logits[0, 1] = -float("inf")

        pos_only_logits = qpn / 0.4
        orig = pos_only_logits[range(4), range(4)].clone()
        correction = torch.log(torch.tensor([1.0, 2.0, 1.0, 2.0]))
        pos_only_logits -= correction
        pos_only_logits[range(4), range(4)] = orig - torch.log(torch.tensor([0.5, 0.5, 1.0, 1.0]))
        pos_only_logits[0, 1] = -float("inf")
        pos_only_logits[1, 0] = -float("inf")
        pos_only_logits[1, 3] = -float("inf")
        pos_only_logits[3, 1] = -float("inf")
        query_ids = torch.tensor([123, 123, 7, 8])
        pos_ids = torch.tensor([64, 7, 8, 7])
        neg_ids = torch.tensor([64])

        softmax_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        return softmax_loss, pos_only_logits, randn_logits, query_ids, pos_ids, neg_ids

    def test_manual_correction_mixed_sum(self):
        for weights in (None, torch.ones(4)):
            with self.subTest(weights=weights):
                (
                    softmax_loss,
                    pos_only_logits,
                    randn_logits,
                    query_ids,
                    pos_ids,
                    neg_ids,
                ) = self.prepare_for_test_manual_correction()
                similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, neg_mode=SoftmaxNegativeMode.MIXED_SUM)
                similarity_loss.temperature = torch.nn.Parameter(torch.tensor(-math.log(0.4)))

                losses = similarity_loss(
                    query=self.query_feat,
                    pos=self.positive_feat,
                    rnd_neg=self.negative_feat,
                    query_ids=query_ids,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    weights=weights,
                )
                expected_rand_loss = softmax_loss(randn_logits, torch.zeros(4, dtype=torch.long))
                expected_pos_loss = softmax_loss(pos_only_logits, torch.arange(4, dtype=torch.long))
                self.assertEqual(len(losses), 2)
                with self.subTest("in batch negatives"):
                    np.testing.assert_array_almost_equal(losses[0].detach(), expected_pos_loss, decimal=3)
                with self.subTest("random negatives"):
                    np.testing.assert_array_almost_equal(losses[1].detach(), expected_rand_loss, decimal=3)

    def test_manual_correction_mixed_concat(self):
        for weights in (None, torch.ones(4)):
            with self.subTest(weights=weights):
                (
                    softmax_loss,
                    pos_only_logits,
                    randn_logits,
                    query_ids,
                    pos_ids,
                    neg_ids,
                ) = self.prepare_for_test_manual_correction()

                similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, neg_mode=SoftmaxNegativeMode.MIXED_CONCAT)
                similarity_loss.temperature = torch.nn.Parameter(torch.tensor(-math.log(0.4)))

                losses = similarity_loss(
                    query=self.query_feat,
                    pos=self.positive_feat,
                    rnd_neg=self.negative_feat,
                    query_ids=query_ids,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    weights=weights,
                )
                expected_loss = softmax_loss(
                    torch.cat([pos_only_logits, randn_logits[:, 1:]], dim=1), torch.arange(4, dtype=torch.long)
                )
                self.assertEqual(len(losses), 1)
                np.testing.assert_array_almost_equal(losses[0].detach(), expected_loss, decimal=3)

    def test_manual_correction_in_batch_only(self):
        for weights in (None, torch.ones(4)):
            with self.subTest(weights=weights):
                (
                    softmax_loss,
                    pos_only_logits,
                    randn_logits,
                    query_ids,
                    pos_ids,
                    neg_ids,
                ) = self.prepare_for_test_manual_correction()

                similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, neg_mode=SoftmaxNegativeMode.IN_BATCH)
                similarity_loss.temperature = torch.nn.Parameter(torch.tensor(-math.log(0.4)))

                losses = similarity_loss(
                    query=self.query_feat,
                    pos=self.positive_feat,
                    query_ids=query_ids,
                    pos_ids=pos_ids,
                    weights=weights,
                )
                expected_loss = softmax_loss(torch.cat([pos_only_logits], dim=1), torch.arange(4, dtype=torch.long))
                self.assertEqual(len(losses), 1)
                np.testing.assert_array_almost_equal(losses[0].detach(), expected_loss, decimal=3)

    def test_manual_correction_random(self) -> None:
        for weights in (None, torch.ones(4)):
            with self.subTest(weights=weights):
                (
                    softmax_loss,
                    pos_only_logits,
                    randn_logits,
                    query_ids,
                    pos_ids,
                    neg_ids,
                ) = self.prepare_for_test_manual_correction()
                similarity_loss = SoftmaxCorrectionLoss(w=256, d=10, neg_mode=SoftmaxNegativeMode.RANDOM)
                similarity_loss.temperature = torch.nn.Parameter(torch.tensor(-math.log(0.4)))

                losses = similarity_loss(
                    query=self.query_feat,
                    pos=self.positive_feat,
                    rnd_neg=self.negative_feat,
                    query_ids=query_ids,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    weights=weights,
                )
                expected_loss = softmax_loss(randn_logits, torch.zeros(4, dtype=torch.long))
                self.assertEqual(len(losses), 1)
                np.testing.assert_array_almost_equal(losses[0].detach(), expected_loss, decimal=3)

    def test_share_mask(self):
        mask = torch.rand((10, 6)) > 0.5
        qid = torch.tensor([1, 3, 2, 1, 3, 1], dtype=torch.long)
        shared_mask = SoftmaxCorrectionLoss._share_masks(mask, qid)
        masks = [
            mask[0] | mask[3] | mask[5],
            mask[2],
            mask[1] | mask[4],
        ]
        expected_mask = torch.stack([masks[q - 1] for q in qid.tolist()])
        np.testing.assert_array_equal(shared_mask, expected_mask)

        # same setting as https://docs.google.com/spreadsheets/d/1E-TAcs6RXUXGAdOpNaAqgg1puj7u_bIs-TbcdWOr5gA/
        qid = torch.tensor([1, 1, 1, 2, 2], dtype=torch.long)
        pid = torch.tensor([1, 2, 3, 1, 4], dtype=torch.long)
        mask = pid.unsqueeze(0) == pid.unsqueeze(1)
        shared_mask = SoftmaxCorrectionLoss._share_masks(mask, qid)
        expected_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        np.testing.assert_array_equal(shared_mask, expected_mask)


class MultiDimSoftmaxCorrectionLossV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.query_feat = torch.tensor([[0.0, 1], [1, 0], [1, 1], [1, 1]])
        self.positive_feat = F.normalize(torch.tensor([[3.0, 3], [11, 11], [14, 14], [1, 0]]), p=2.0, dim=1)
        self.negative_feat = F.normalize(torch.tensor([[1.0, 2]]), p=2.0, dim=1)

    def test_repr(self):
        self.maxDiff = 1000
        self.assertEqual(
            str(MultiDimSoftmaxCorrectionLossV2(synchronize_pos=True, d=3, w=24, emb_dim_weights={32: 0.5, 256: 1})),
            """MultiDimSoftmaxCorrectionLossV2(
  neg_mode=MIXED_SUM, enable_spc=True, max_in_batch_negatives=None, emb_dim_weights=32:0.5,256:1
  (conditional_ct): ConditionalProbabilityEstimator(
    (q_freq_est): CountMinSketch(w=24,d=3)
    (qp_freq_est): CountMinSketch(w=24,d=3)
  )
  (in_batch_neg_ct): NegativeCounter(
    (freq_est): CountMinSketch(w=24,d=3)
  )
  (rnd_neg_ct): NegativeCounter(
    (freq_est): CountMinSketch(w=24,d=3)
  )
)""",
        )

    def test_forward(self):
        query_ids = torch.tensor([1, 3, 7, 9])
        pos_ids = torch.tensor([3, 7, 1, 1])
        neg_ids = torch.tensor([4])
        similarity_loss = MultiDimSoftmaxCorrectionLossV2(
            w=256, d=10, synchronize_pos=True, emb_dim_weights={1: 0.5, 2: 1}
        )
        losses = similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=query_ids,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        with self.subTest("first call"):
            np.testing.assert_array_almost_equal(
                torch.stack(losses).detach(), [0.5493, 0.3466, 2.6161, 1.6603], decimal=3
            )

        # update cms table
        similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=torch.tensor([7, 7, 9, 7]),
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        losses = similarity_loss(
            query=self.query_feat,
            pos=self.positive_feat,
            rnd_neg=self.negative_feat,
            query_ids=query_ids,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
        )
        with self.subTest("third call"):
            np.testing.assert_array_almost_equal(
                torch.stack(losses).detach(), [0.5105, 0.3187, 2.5385, 1.6256], decimal=3
            )


if __name__ == "__main__":
    unittest.main()
