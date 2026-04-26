# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestDispatchCount(mlx_tests.MLXTestCase):
    def test_counts_increase_then_reset(self):
        if not mx.metal.is_available():
            return

        mx.metal.reset_dispatch_count()
        self.assertEqual(mx.metal.dispatch_count(), 0)

        a = mx.array([1.0, 2.0, 3.0])
        b = a + 1.0
        mx.eval(b)

        after_one_op = mx.metal.dispatch_count()
        self.assertGreater(after_one_op, 0)

        c = b * 2.0
        mx.eval(c)
        self.assertGreater(mx.metal.dispatch_count(), after_one_op)

        mx.metal.reset_dispatch_count()
        self.assertEqual(mx.metal.dispatch_count(), 0)


if __name__ == "__main__":
    unittest.main()
