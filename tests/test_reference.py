import unittest

import numpy as np

from examples.boundary import scan
from woodbury_reference import dense_loglike, woodbury_loglike


class ReferenceTests(unittest.TestCase):
    def test_boundary_scan(self):
        grid, curves = scan()
        np.testing.assert_allclose(curves[0], curves[1], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(grid[curves.argmax(axis=1)], [0., 0.])

    def test_zero_column_and_all_zero(self):
        d = np.array([[2., .25], [.25, 1.]])
        u = np.array([[1., .5], [.5, 1.]])
        r = np.array([2., -1.])
        for fn in (dense_loglike, woodbury_loglike):
            self.assertEqual(fn(d, u, [1, 0], r), fn(d, u[:, :1], [1], r))
            expected = -.5 * (r @ np.linalg.solve(d, r) + np.linalg.slogdet(d)[1]
                              + len(r) * np.log(2*np.pi))
            self.assertAlmostEqual(fn(d, u, [0, 0], r), expected, places=12)

    def test_invalid_weights_and_inactive_column(self):
        d = np.eye(2)
        for fn in (dense_loglike, woodbury_loglike):
            for u, w in ((d, [-1, 0]), (d, [0, np.inf]),
                         (np.array([[1, np.nan], [0, 1]]), [1, 0])):
                with self.subTest(method=fn.__name__, weights=w):
                    with self.assertRaises(ValueError):
                        fn(d, u, w, [1, -1])
