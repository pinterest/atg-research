from __future__ import annotations

import unittest

import torch
from omnisearchsage.modules.approx import CountMinSketch


class CountMinSketchTest(unittest.TestCase):
    def test_repr(self):
        self.maxDiff = 2000
        module = CountMinSketch(w=12, d=23)

        expected = "CountMinSketch(w=12,d=23)"
        self.assertEqual(repr(module), expected)

    def test_collisions(self):
        torch.manual_seed(0)
        sketch: CountMinSketch = CountMinSketch(w=2**2, d=1, seed=12321)

        input_ids = torch.randint(1000000, (50,))
        sketch.update(input_ids)
        min_cts, total = sketch(input_ids)
        self.assertGreaterEqual(min_cts.max().item(), 50 / 4)
        self.assertEqual(50, total.item())

    def test_update_larger(self):
        torch.manual_seed(0)
        sketch: CountMinSketch = CountMinSketch(w=2**10, d=8, seed=12321)

        input_ids = torch.randint(1000000, (50,))
        cts = torch.randint(1000, (50,))
        sketch.update(input_ids.repeat_interleave(cts, dim=0), increment=1)
        approx_cts, total = sketch(input_ids)
        self.assertTrue((approx_cts >= cts).all())
        self.assertEqual(cts.sum().item(), total.item())


if __name__ == "__main__":
    unittest.main()
