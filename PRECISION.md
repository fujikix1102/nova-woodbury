# Precision limits

The implementation uses binary64. The Woodbury quadratic subtracts two
terms, so cancellation can be significant for extreme inputs. For the
scalar case D=U=1, b=256 and r=2^20, an internal audit recorded an absolute
log-likelihood error of about 5.36e-5 against a high-precision reference.
There is no automatic accuracy guarantee or fallback.

Internal same-Mac tests of the identical evaluator source covered 72
synthetic precision cases and a 4,320-point synthetic boundary grid.
These tests establish agreement on the tested cases. The example here
is a smaller standalone regression check. Performance, external use
and real observational inputs have not been evaluated for this draft.

The larger internal audits were repeated with the installed evaluator
on NumPy 2.5.4. Both methods passed all 72 precision cases. Maximum
absolute errors against the 90-digit reference were 1.7458e-13 for dense
and 1.5046e-13 for Woodbury. The 4,320-point boundary grid included
80 zero-weight and 4,240 positive-weight points; maximum errors were
4.2013e-15 and 4.2533e-15, respectively. The grid maximum was (3.625, 0)
for both methods. The same numerical rows and reported errors were
observed with NumPy 2.2.4. Neither version has a general error guarantee.
