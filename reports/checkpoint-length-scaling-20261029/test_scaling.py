import unittest
import numpy as np
from analyze_scaling import design, nonnegative_fit


class ScalingTests(unittest.TestCase):
    def test_design_distinguishes_same_total_tokens_different_lengths(self):
        first, second = design(16, 1024), design(8, 2048)
        self.assertEqual(first[1], second[1])
        self.assertNotEqual(first[2], second[2])

    def test_fit_recovers_positive_linear_and_quadratic_terms(self):
        x = np.asarray([design(b, length) for b in (8, 16, 32) for length in (256, 512, 1024, 2048)])
        expected = np.asarray([1.1, 3.2, .4])
        fitted, error = nonnegative_fit(x, x @ expected, [0, 1, 2])
        np.testing.assert_allclose(fitted, expected, atol=1e-12)
        self.assertLess(error, 1e-12)


if __name__ == "__main__":
    unittest.main()
