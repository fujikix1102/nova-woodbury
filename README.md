# NOVA Woodbury reference

A NumPy reference for normalized Gaussian log likelihoods with covariance
D + U diag(b) Uᵀ. Compare dense Cholesky with the Woodbury identity and
scan update strengths including a zero-weight boundary.

Requires Python 3.11+ and NumPy 2.0+. From a local source copy, create and
activate a fresh virtual environment, then run:

    python -m pip install .
    python examples/boundary.py
    python -m unittest discover -s tests -v

D must be exactly symmetric and positive definite. The input basis must
have at least one column and all inputs must be finite, including inactive
columns. Weights must be nonnegative. All-zero weights evaluate D alone.
See [precision limits](PRECISION.md) for the validation scope.

## Verified environments

Same Apple M1 Pro, Python 3.12.12. The identical evaluator source was
checked in the following environments.

| NumPy | 72 synthetic precision cases | 4,320-point boundary grid |
|---|---|---|
| 2.2.4 | Passed | Passed |
| 2.5.4 | Passed | Passed |

The 72 cases are regenerated from a recorded seed and procedure. The
boundary grid uses frozen arrays. Numerical rows, errors and boundary
decisions matched between these versions. The JSON files differ in the
recorded NumPy version. The small standalone example and three tests
also passed after ordinary local-source installation with NumPy 2.5.4.
The larger internal audits use additional materials outside this draft.
These observations cover the tested inputs and software combinations.

## Related paper

Junichi Fujiki (2026). *Guarded Low-Rank Woodbury Covariance Updates for Reproducible Cosmological Likelihood Evaluation*. Zenodo.
[https://doi.org/10.5281/zenodo.24181679](https://doi.org/10.5281/zenodo.24181679)

This repository provides a standalone NumPy reference implementation for the low-rank covariance updates discussed in the paper.
