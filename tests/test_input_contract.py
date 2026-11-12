import unittest
import numpy as np
from woodbury_reference import dense_loglike, woodbury_loglike


class InputContractTests(unittest.TestCase):
    def test_rejected_inputs(self):
        d = np.eye(2)
        u = np.ones((2, 1))
        cases = [
            ("nonsquare", np.ones((2, 3)), u, [1], [0, 0]),
            ("empty_base", np.empty((0, 0)), np.empty((0, 1)), [1], []),
            ("basis_rows", d, np.ones((3, 1)), [1], [0, 0]),
            ("empty_columns", d, np.empty((2, 0)), [], [0, 0]),
            ("weight_shape", d, u, [[1]], [0, 0]),
            ("residual_shape", d, u, [1], [[0, 0]]),
            ("asymmetric", np.array([[1., .1], [0., 1.]]), u, [1], [0, 0]),
            ("indefinite", np.diag([1., -1.]), np.array([[0.], [2.]]), [1], [0, 0]),
            ("singular", np.diag([1., 0.]), np.array([[0.], [2.]]), [1], [0, 0]),
            ("nonfinite_base", np.diag([1., np.inf]), u, [1], [0, 0]),
            ("nonfinite_basis", d, np.array([[np.nan], [1.]]), [1], [0, 0]),
            ("nonfinite_residual", d, u, [1], [0, np.inf]),
            ("nan_weight", d, u, [np.nan], [0, 0]),
            ("negative_weight", d, u, [-1], [0, 0]),
        ]
        for method in (dense_loglike, woodbury_loglike):
            for label, *args in cases:
                with self.subTest(api=method.__name__, case=label):
                    with self.assertRaises(ValueError):
                        method(*args)

    def test_zero_weights_still_validate_inputs(self):
        for method in (dense_loglike, woodbury_loglike):
            for base, basis, residual in (
                (np.eye(2), np.array([[np.inf], [1.]]), [0, 0]),
                (np.eye(2), np.ones((2, 1)), [np.nan, 0]),
                (np.diag([1., -1.]), np.ones((2, 1)), [0, 0]),
            ):
                with self.subTest(api=method.__name__, base=base.tolist()):
                    with self.assertRaises(ValueError):
                        method(base, basis, [0], residual)

    def test_zero_weight_boundary_agrees_with_dense(self):
        base = np.array([[2., .5], [.5, 1.]])
        basis = np.array([[1., .25], [-.5, 1.]])
        residual = np.array([.2, -.3])
        for weights in ([0., 0.], [0., 1.], [1., 0.]):
            with self.subTest(weights=weights):
                self.assertAlmostEqual(
                    dense_loglike(base, basis, weights, residual),
                    woodbury_loglike(base, basis, weights, residual),
                    places=12,
                )
