"""
signature.py
============

From-scratch, vectorized implementation of the truncated path signature,
batched over Monte Carlo paths, built directly on Chen's identity. No
external signature library (iisignature / esig): the point of this module
is to keep the algebra visible, since it's the object the rest of the
project (hedge_experiment.py) is built on.

Background
----------
For a continuous path X : [0, T] -> R^d, the level-k iterated integral is

    S^k(X)_{0,T} = \\int_{0<t_1<...<t_k<T} dX_{t_1} \\otimes ... \\otimes dX_{t_k}
                 \\in (R^d)^{\\otimes k}

and the signature is the collection Sig(X) = (1, S^1, S^2, S^3, ...). It's
universal / non-parametric: any continuous function of the path can be
approximated arbitrarily well by a linear function of Sig(X) (Lyons, McLeod
2022, Signature Methods in Machine Learning, arXiv:2206.14674). That's the
property we lean on later: fit a hedge ratio that's linear in the signature
and still capture genuinely path-dependent, nonlinear-in-price behaviour.

For a piecewise-linear path (exactly what a discretized simulated price
path is), the signature of a single segment with increment delta in R^d is
the truncated tensor exponential

    exp^{<=M}(delta) = sum_{k=0}^{M} delta^{\\otimes k} / k!

and the signature of the concatenation of two paths is the tensor product
(Chen's identity):

    Sig(X * Y) = Sig(X) (x) Sig(Y)

So the running signature of a discretized path is built by repeatedly
"multiplying in" the tensor exponential of each new increment.

Implementation
---------------
Level-k tensors in (R^d)^{\\otimes k} are stored flattened as vectors in
R^{d^k}, row-major: the flat index of (i_1, ..., i_k) is
i_1 * d^{k-1} + i_2 * d^{k-2} + ... + i_k. Two flattened tensors A in
R^{d^i} and B in R^{d^j} combine into their tensor product A (x) B in
R^{d^{i+j}} via

    (A (x) B)_flat = flatten( outer(A, B) ) = (A[:, None] * B[None, :]).ravel()

which matches the recursive flattening above. Everything carries a leading
batch axis N (one row per Monte Carlo path), so the full truncated
signature of N paths with n steps each comes out in O(n) vectorized numpy
operations, no per-path Python loop.
"""

from __future__ import annotations
import numpy as np
from math import factorial


def _batched_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched flattened tensor product. A: (N, dA), B: (N, dB) -> (N, dA*dB)."""
    N = A.shape[0]
    return (A[:, :, None] * B[:, None, :]).reshape(N, -1)


class TruncatedSignature:
    """
    Keeps the running truncated signature (levels 0..M) of a batch of
    d-dimensional piecewise-linear paths, fed one increment at a time.

    This streaming form is what a hedging strategy actually needs: the
    hedge ratio at t_i has to be a function of the signature of the path
    up to t_i only (adaptedness, no look-ahead).
    """

    def __init__(self, n_paths: int, dim: int, level: int):
        self.N = n_paths
        self.d = dim
        self.M = level
        # S[k] has shape (N, d**k); S[0] is the constant 1 (empty word).
        self.S = [np.ones((n_paths, 1))]
        for k in range(1, level + 1):
            self.S.append(np.zeros((n_paths, dim ** k)))

    def step(self, increments: np.ndarray) -> None:
        """
        Update the running signature with one batch of increments.
        increments: (N, d), delta_t = X_{t+1} - X_t for every path.
        """
        N, d, M = self.N, self.d, self.M

        # E[k] = increments^{(x)k} / k!  (tensor exponential of the step)
        E = [np.ones((N, 1))]
        cur = np.ones((N, 1))
        for k in range(1, M + 1):
            cur = _batched_outer(cur, increments)  # (N, d**k), unnormalized
            E.append(cur / factorial(k))

        # Chen's identity: new S[k] = sum_{i=0}^{k} S[i] (x) E[k-i]
        new_S = [None] * (M + 1)
        new_S[0] = self.S[0]  # constant term stays 1
        for k in range(1, M + 1):
            acc = np.zeros((N, d ** k))
            for i in range(0, k + 1):
                acc += _batched_outer(self.S[i], E[k - i])
            new_S[k] = acc
        self.S = new_S

    def features(self) -> np.ndarray:
        """Concatenate levels 1..M into one flat feature vector per path (drop the trivial level-0 '1')."""
        return np.concatenate(self.S[1:], axis=1)

    def feature_dim(self) -> int:
        return sum(self.d ** k for k in range(1, self.M + 1))


def running_signature_features(paths: np.ndarray, level: int) -> np.ndarray:
    """
    Full history of running signature feature vectors for a batch of paths.

    paths : (N, n+1, d), N paths, n+1 time points, d channels
            (channel 0 is time by convention, see hedge_experiment.py)
    level : truncation level M

    Returns
    -------
    feats : (N, n+1, sig_dim), feats[:, i, :] is the signature (levels
            1..M) of the path restricted to [0, t_i], i.e. using only
            information available at t_i (adapted, no look-ahead).
            feats[:, 0, :] is all zeros (signature of a single point).
    """
    N, n_plus_1, d = paths.shape
    n = n_plus_1 - 1
    sig = TruncatedSignature(N, d, level)
    sig_dim = sig.feature_dim()
    feats = np.zeros((N, n_plus_1, sig_dim))
    for i in range(n):
        inc = paths[:, i + 1, :] - paths[:, i, :]
        sig.step(inc)
        feats[:, i + 1, :] = sig.features()
    return feats


if __name__ == "__main__":
    # Sanity check against a hand-computable case: for a 1D path (d=1)
    # that is simply t on [0,1] split into n equal steps, the level-k
    # term of the signature must equal t^k / k! (the signature of a
    # straight line is its own tensor exponential).
    n = 200
    t = np.linspace(0, 1, n + 1).reshape(1, n + 1, 1)
    feats = running_signature_features(t, level=4)
    got = feats[0, -1, :]  # levels 1..4 at t=1 for the single path
    expected = np.array([1 / factorial(k) for k in range(1, 5)])
    print("numerical running-signature levels 1..4 at t=1:", got)
    print("closed-form t^k/k! at t=1:                      ", expected)
    print("max abs error:", np.max(np.abs(got - expected)))
