"""
nn.py
======

A small, from-scratch feedforward neural network and Adam optimizer.
It is numpy-only, with manual backprop.

No autodiff framework is used: every forward and backward formula
is written out explicitly.
This is for transparency,
and also because the deep-hedging training loop in `deep_hedging.py`
must backpropagate gradients THROUGH the wealth accumulation across time steps
and INTO a separate small network at each rebalancing date.
That is easiest to get right (and to debug) when every piece
of the chain rule is visible.

Architecture per network: Dense -> tanh -> Dense -> tanh -> Dense -> sigmoid
(2 hidden layers).

The final sigmoid squashes the output hedge ratio into [0, 1].
This is a deliberate modeling choice:
for a vanilla call, any sensible hedge ratio lives in [0, 1]
(like a Black-Scholes delta),
and constraining the output range both regularizes training
and keeps the learned strategy economically interpretable.
"""

from __future__ import annotations
import numpy as np


def _glorot(rng, n_in, n_out):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


# One Dense layer + tanh activation, with manual forward/backward.
class DenseTanh:

    def __init__(self, n_in, n_out, rng):
        self.W = _glorot(rng, n_in, n_out)
        self.b = np.zeros(n_out)
        self.cache = None

    def forward(self, X):
        z = X @ self.W + self.b
        a = np.tanh(z)
        self.cache = (X, a)
        return a

    def backward(self, dA):
        X, a = self.cache
        dz = dA * (1.0 - a ** 2)
        dW = X.T @ dz
        db = dz.sum(axis=0)
        dX = dz @ self.W.T
        return dX, dW, db


# Final Dense layer + sigmoid activation (output squashed to [0, 1]).
class DenseSigmoid:

    def __init__(self, n_in, n_out, rng):
        self.W = _glorot(rng, n_in, n_out)
        self.b = np.zeros(n_out)
        self.cache = None

    def forward(self, X):
        z = X @ self.W + self.b
        a = 1.0 / (1.0 + np.exp(-z))
        self.cache = (X, a)
        return a

    def backward(self, dA):
        X, a = self.cache
        dz = dA * a * (1.0 - a)
        dW = X.T @ dz
        db = dz.sum(axis=0)
        dX = dz @ self.W.T
        return dX, dW, db


# One small MLP mapping a market-state feature vector
# (time-to-maturity, log-moneyness) to a scalar hedge ratio in [0, 1].
# It is used with INDEPENDENT weights at every rebalancing date,
# as in Buehler, Gonon, Teichmann & Wood 2019
# (original, non-recurrent architecture).
#
# Deliberately NOT fed the previous holding theta_{i-1} as an input.
# Doing so would make theta_i a function of theta_{i-1}
# through the FORWARD computation graph,
# so d(loss)/d(theta_i) would depend on every LATER time step's network too.
# That is a genuine recurrence, requiring full backprop-through-time.
#
# Keeping the state purely Markovian in (t, S_t)
# keeps every network's gradient local to its own two adjacent
# transaction-cost terms (see the backward pass in deep_hedging.py).
# This is a deliberate scope choice, noted as a limitation in the README:
# the natural extension is adding theta_{i-1} back in,
# together with full BPTT through the resulting recurrence.
class HedgeNet:

    def __init__(self, n_features: int, n_hidden: int, seed: int):
        rng = np.random.default_rng(seed)
        self.l1 = DenseTanh(n_features, n_hidden, rng)
        self.l2 = DenseTanh(n_hidden, n_hidden, rng)
        self.l3 = DenseSigmoid(n_hidden, 1, rng)
        self.layers = [self.l1, self.l2, self.l3]

    def forward(self, X):
        a = self.l1.forward(X)
        a = self.l2.forward(a)
        a = self.l3.forward(a)
        return a[:, 0]  # (N,)

    def backward(self, dtheta):
        dA = dtheta[:, None]
        dA, dW3, db3 = self.l3.backward(dA)
        dA, dW2, db2 = self.l2.backward(dA)
        _, dW1, db1 = self.l1.backward(dA)
        return [(dW1, db1), (dW2, db2), (dW3, db3)]

    def params(self):
        return [(l.W, l.b) for l in self.layers]

    def set_params(self, params):
        for l, (W, b) in zip(self.layers, params):
            l.W, l.b = W, b


# Standard Adam optimizer (Kingma & Ba 2014),
# applied to a flat list of (param_array, grad_array) pairs.
# It is generic enough to be reused across every HedgeNet in the model,
# plus the scalar indifference price p0.
class Adam:

    def __init__(self, shapes, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            new_params.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return new_params


if __name__ == "__main__":
    # Gradient check: compare the analytic backward pass of a HedgeNet
    # against a central finite-difference estimate,
    # for a scalar loss built directly on its output.
    # This validates the hand-coded backprop chain
    # BEFORE it is trusted inside the much more complex
    # hedging training loop of deep_hedging.py.
    rng = np.random.default_rng(0)
    net = HedgeNet(n_features=3, n_hidden=6, seed=1)
    X = rng.normal(size=(5, 3))

    def loss_fn(net):
        out = net.forward(X)
        return np.sum(out ** 2)  # arbitrary scalar loss

    out = net.forward(X)
    dtheta = 2 * out  # d(loss)/d(out)
    grads = net.backward(dtheta)

    eps = 1e-6
    max_rel_err = 0.0
    for layer_idx, (l, (dW, db)) in enumerate(zip(net.layers, grads)):
        # check a handful of weight entries, not the whole matrix (speed)
        idxs = [(i, j)
                for i in range(min(3, l.W.shape[0]))
                for j in range(min(3, l.W.shape[1]))]
        for (i, j) in idxs:
            orig = l.W[i, j]
            l.W[i, j] = orig + eps
            lp = loss_fn(net)
            l.W[i, j] = orig - eps
            lm = loss_fn(net)
            l.W[i, j] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = dW[i, j]
            rel_err = abs(numeric - analytic) / max(abs(numeric), 1e-8)
            max_rel_err = max(max_rel_err, rel_err)
    print(f"Gradient check (weights, {len(net.layers)} layers, "
          f"few entries each): max relative error = {max_rel_err:.2e}")
    assert max_rel_err < 1e-4, "backprop gradient check FAILED"
    print("PASSED")
