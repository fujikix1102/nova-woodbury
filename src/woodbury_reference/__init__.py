"""Gaussian log likelihoods for nonnegative low-rank covariance updates."""
import numpy as np


def _inputs(base, basis, weights, residual):
    d, u, b, r = (np.asarray(x, dtype=np.float64)
                  for x in (base, basis, weights, residual))
    if d.ndim != 2 or not d.shape[0] or d.shape[0] != d.shape[1]:
        raise ValueError("base must be nonempty and square")
    n = d.shape[0]
    if (u.ndim != 2 or u.shape[0] != n or not u.shape[1]
            or b.shape != (u.shape[1],) or r.shape != (n,)):
        raise ValueError("incompatible basis, weights or residual")
    if not all(np.isfinite(x).all() for x in (d, u, b, r)):
        raise ValueError("inputs must be finite")
    if not np.array_equal(d, d.T) or np.any(b < 0):
        raise ValueError("base must be symmetric and weights nonnegative")
    active = b > 0
    return d, u[:, active], b[active], r


def _factor(matrix):
    if not np.isfinite(matrix).all():
        raise ValueError("covariance calculation overflowed")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix must be positive definite") from exc


def _normalized(quadratic, logdet, n):
    value = -0.5 * (quadratic + logdet + n * np.log(2 * np.pi))
    if not np.isfinite(value):
        raise ValueError("nonfinite log likelihood")
    return float(value)


def dense_loglike(base, basis, weights, residual):
    """Form the full updated covariance and evaluate by Cholesky."""
    d, u, b, r = _inputs(base, basis, weights, residual)
    _factor(d)
    factor = _factor(d + (u * b) @ u.T)
    y = np.linalg.solve(factor, r)
    return _normalized(y @ y, 2 * np.log(factor.diagonal()).sum(), r.size)


def woodbury_loglike(base, basis, weights, residual):
    """Use the whitened Woodbury identity and determinant lemma."""
    d, u, b, r = _inputs(base, basis, weights, residual)
    factor = _factor(d)
    y = np.linalg.solve(factor, r)
    if not b.size:
        return _normalized(y @ y, 2 * np.log(factor.diagonal()).sum(), r.size)
    v = np.linalg.solve(factor, u * np.sqrt(b))
    reduced = _factor(np.eye(b.size) + v.T @ v)
    correction = np.linalg.solve(reduced, v.T @ y)
    quadratic = y @ y - correction @ correction
    logdet = 2 * (np.log(factor.diagonal()).sum()
                  + np.log(reduced.diagonal()).sum())
    return _normalized(quadratic, logdet, r.size)
