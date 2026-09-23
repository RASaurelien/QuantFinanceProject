"""
deep_hedging.py
=================

Deep Hedging (Buehler, Gonon, Teichmann & Wood, 2019, "Deep Hedging",
Quantitative Finance 19(8), arXiv:1802.03042).

We train a hedging strategy:
one small neural network per rebalancing date,
each mapping the local market state to a hedge ratio in [0, 1].
The goal is to directly minimize a CONVEX RISK MEASURE
of the terminal hedging error,
under REALISTIC proportional transaction costs.

This is the setting where a classical closed-form Greek
(the Black-Scholes delta) is not even the right OBJECT to compute.
Transaction costs make the optimal hedge depend on the trading history itself
(via the turnover cost),
which is exactly why a learned, flexible policy
has room to add value over a fixed formula.

Risk measure: the ENTROPIC risk measure of the hedging error e,

    rho_lambda(e) = (1/lambda) * log( E[ exp(lambda * e) ] )

with e_j = payoff_j - (p0 + hedge_gains_j - transaction_costs_j),
the amount the hedger still owes at maturity,
net of the premium collected and the hedging P&L.

rho_lambda is convex, monotone and translation invariant.
As lambda -> 0 it converges to E[e] (risk-neutral).
For lambda > 0 it increasingly penalizes the UPPER TAIL of e
(states where the hedge badly under-covers the payoff),
which is exactly the tail risk a hedger cares about.

The premium p0 is FIXED (Monte Carlo fair value
under the simulated risk-neutral dynamics), not jointly optimized.
Minimizing rho_lambda over p0 as well would be a degenerate, unbounded direction
(d(rho)/dp0 = -1 identically, i.e. "charge an infinite premium").
So only the certainty-equivalent formulation of Buehler et al.
(fix p0, optimize the hedge) is implemented here.

Backpropagation through time
------------------------------
Networks are NOT weight-shared across time steps
(following the original paper's architecture),
so gradients never flow from one network's weights into another's.

They DO have to flow backward through the TRANSACTION COST term,
which links theta_i to BOTH cost_i (via the trade theta_i - theta_{i-1})
and cost_{i+1} (via the NEXT trade theta_{i+1} - theta_i).

This inter-step coupling is handled explicitly in `loss_and_grad()` below.
For every time step i (processed in any order, since networks are independent),
we accumulate the two contributions to d(loss)/d(theta_i):
one from cost_i, one from cost_{i+1}.
"""

from __future__ import annotations
import numpy as np

from nn import HedgeNet, Adam


class DeepHedge:
    def __init__(self, n_steps: int, n_hidden: int = 16, seed: int = 0):
        self.n_steps = n_steps
        self.nets = [HedgeNet(n_features=2, n_hidden=n_hidden, seed=seed + i)
                     for i in range(n_steps)]

    # ------------------------------------------------------------------
    # Inputs:
    #   t : (n_steps+1,) time grid
    #   S : (N, n_steps+1) price paths
    #
    # Returns: theta (N, n_steps), wealth (N,),
    # plus the caches needed for the backward pass.
    def forward(self, t, S, K, cost_rate):
        N = S.shape[0]
        T = t[-1]
        theta_prev = np.zeros(N)
        theta_hist = np.zeros((N, self.n_steps))
        wealth = np.zeros(N)
        state_cache = []

        for i in range(self.n_steps):
            tau = (T - t[i]) / T  # time-to-maturity, normalized to [0, 1]
            logm = np.log(S[:, i] / K)
            X = np.stack([np.full(N, tau), logm], axis=1)
            theta_i = self.nets[i].forward(X)

            dS = S[:, i + 1] - S[:, i]
            cost_i = cost_rate * np.abs(theta_i - theta_prev) * S[:, i]
            wealth += theta_i * dS - cost_i

            theta_hist[:, i] = theta_i
            state_cache.append(X)
            theta_prev = theta_i

        return theta_hist, wealth, state_cache

    # ------------------------------------------------------------------
    # Forward pass + entropic-risk loss + full backward pass,
    # through the wealth/cost accumulation AND into every per-step network.
    #
    # Returns (loss, grads), where grads is a list (one entry per time step)
    # of [(dW1, db1), (dW2, db2), (dW3, db3)].
    def loss_and_grad(self, t, S, K, cost_rate, payoff, p0, lam):
        N, n = S.shape[0], self.n_steps
        theta_hist, wealth, state_cache = self.forward(t, S, K, cost_rate)
        err = payoff - (p0 + wealth)  # hedging error (positive = hedge falls short)

        # ---- entropic risk loss & its gradient wrt err (softmax weights) ----
        w = np.exp(lam * (err - err.max()))  # subtract max for numerical stability
        M = w.mean()
        loss = err.max() + np.log(M) / lam
        dLoss_derr = w / (N * M)  # (N,), sums to 1

        # d(err)/d(wealth) = -1, so d(loss)/d(wealth contribution) = -dLoss_derr
        dL_dwealth = -dLoss_derr  # (N,)

        # ---- backward through the wealth/cost accumulation ----
        # d(wealth)/d(theta_i) = dS_i - d(cost_i)/d(theta_i) - d(cost_{i+1})/d(theta_i)
        S_grid = S

        # theta_{i-1}, including theta_{-1} = 0
        theta_prev_arr = np.concatenate(
            [np.zeros((N, 1)), theta_hist[:, :-1]], axis=1)
        # theta_{i+1}, last column = NaN (no next trade)
        theta_next_arr = np.concatenate(
            [theta_hist[:, 1:], np.full((N, 1), np.nan)], axis=1)

        grads = []
        for i in range(n):
            dS_i = S_grid[:, i + 1] - S_grid[:, i]
            theta_i = theta_hist[:, i]
            theta_im1 = theta_prev_arr[:, i]

            # d(cost_i)/d(theta_i)
            d_cost_i = cost_rate * np.sign(theta_i - theta_im1) * S_grid[:, i]

            if i < n - 1:
                theta_ip1 = theta_next_arr[:, i]
                # d(cost_{i+1})/d(theta_i)
                d_cost_ip1 = (-cost_rate * np.sign(theta_ip1 - theta_i)
                              * S_grid[:, i + 1])
            else:
                d_cost_ip1 = 0.0

            d_wealth_d_theta_i = dS_i - d_cost_i - d_cost_ip1
            dL_dtheta_i = dL_dwealth * d_wealth_d_theta_i  # chain rule, (N,)

            grads.append(self.nets[i].backward(dL_dtheta_i))

        return loss, grads

    # ------------------------------------------------------------------
    def train(self, t, S, K, cost_rate, payoff, p0, lam,
              n_epochs, batch_size, lr, verbose_every=0):
        N = S.shape[0]
        shapes = []
        for net in self.nets:
            for (W, b) in net.params():
                shapes.append(W.shape)
                shapes.append(b.shape)
        opt = Adam(shapes, lr=lr)

        loss_history = []
        rng = np.random.default_rng(123)
        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_losses = []
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                loss, grads = self.loss_and_grad(
                    t, S[idx], K, cost_rate, payoff[idx], p0, lam)
                epoch_losses.append(loss)

                # flatten params/grads across all nets and layers,
                # step Adam, then unflatten back
                flat_params, flat_grads = [], []
                for net_grads, net in zip(grads, self.nets):
                    for (W, b), (dW, db) in zip(net.params(), net_grads):
                        flat_params.append(W)
                        flat_grads.append(dW)
                        flat_params.append(b)
                        flat_grads.append(db)
                new_flat = opt.step(flat_params, flat_grads)

                k = 0
                for net in self.nets:
                    new_params = []
                    for _ in net.params():
                        new_params.append((new_flat[k], new_flat[k + 1]))
                        k += 2
                    net.set_params(new_params)

            loss_history.append(np.mean(epoch_losses))
            if verbose_every and (epoch % verbose_every == 0
                                  or epoch == n_epochs - 1):
                print(f"  epoch {epoch:4d}  "
                      f"entropic-risk loss = {loss_history[-1]:.5f}")
        return loss_history


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Full-pipeline gradient check.
    #
    # We finite-difference the TOTAL loss
    # (forward -> entropic risk, through the wealth/cost accumulation,
    # into ONE specific weight of ONE specific time-step's network)
    # and compare it to the analytic backward pass.
    #
    # This is the check that matters most:
    # the inter-timestep transaction-cost coupling
    # (not exercised by nn.py's standalone gradient check)
    # is the easiest part to get wrong.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n_steps = 6
    N = 40
    t = np.linspace(0, 0.25, n_steps + 1)
    S0, K = 100.0, 100.0

    # A simple toy GBM path set.
    # This file does not need rough_bergomi to validate the backprop:
    # any price path array will do.
    dt = t[1] - t[0]
    sigma = 0.2
    dW = rng.normal(size=(N, n_steps)) * np.sqrt(dt)
    logS = np.log(S0) + np.cumsum(-0.5 * sigma ** 2 * dt + sigma * dW, axis=1)
    S = np.concatenate([np.full((N, 1), S0), np.exp(logS)], axis=1)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    cost_rate = 0.01
    p0 = payoff.mean()
    lam = 5.0

    model = DeepHedge(n_steps=n_steps, n_hidden=5, seed=1)
    loss0, grads = model.loss_and_grad(t, S, K, cost_rate, payoff, p0, lam)

    # Pick a weight in the network of an INTERIOR time step
    # (the most exposed to the two-sided cost coupling)
    # and finite-difference it.
    i_step = n_steps // 2
    net = model.nets[i_step]
    layer = net.l1
    eps = 1e-6
    max_rel_err = 0.0
    for (r, c) in [(0, 0), (1, 2), (2, 1)]:
        if r >= layer.W.shape[0] or c >= layer.W.shape[1]:
            continue
        orig = layer.W[r, c]
        layer.W[r, c] = orig + eps
        lp, _ = model.loss_and_grad(t, S, K, cost_rate, payoff, p0, lam)
        layer.W[r, c] = orig - eps
        lm, _ = model.loss_and_grad(t, S, K, cost_rate, payoff, p0, lam)
        layer.W[r, c] = orig
        numeric = (lp - lm) / (2 * eps)
        analytic = grads[i_step][0][0][r, c]
        rel_err = abs(numeric - analytic) / max(abs(numeric), 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  weight ({r},{c}) of step {i_step}: "
              f"numeric={numeric:.6e}  analytic={analytic:.6e}  "
              f"rel_err={rel_err:.2e}")

    print("\nFull-pipeline gradient check "
          f"(through transaction-cost coupling): max relative error = {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, "full-pipeline backprop gradient check FAILED"
    print("PASSED")
