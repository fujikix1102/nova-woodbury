"""Find a synthetic covariance-update maximum on a grid including zero."""
import numpy as np

from woodbury_reference import dense_loglike, woodbury_loglike


def scan():
    base = np.array([[1., .25], [.25, 1.]])
    basis = np.array([[1.], [1.]])
    residual = np.array([1., -1.])
    grid = np.linspace(0, 2, 21)
    curves = np.array([[fn(base, basis, [b], residual) for b in grid]
                       for fn in (dense_loglike, woodbury_loglike)])
    return grid, curves


if __name__ == "__main__":
    grid, curves = scan()
    np.testing.assert_allclose(curves[0], curves[1], atol=1e-9, rtol=0)
    print("Grid maxima:", grid[curves.argmax(axis=1)])
    print("Maximum absolute log-likelihood difference:", np.max(np.abs(curves[0]-curves[1])))
