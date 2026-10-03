"""One GP and EI implementation, with optional Gaussian-gated prior basis functions."""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bo_eval.domain import Domain
from bo_eval.graph import Node, Prior


@dataclass
class Posterior:
    mean: np.ndarray
    sd: np.ndarray
    gates: dict[str, float] = field(default_factory=dict)
    gate_sd: dict[str, float] = field(default_factory=dict)


def prior_weight(env: Domain, candidates: np.ndarray, belief: dict) -> np.ndarray:
    """A Gaussian bump in encoded coordinates; widths are fractions of that range."""
    logp = np.zeros(len(candidates))
    for param, (best, width) in belief.items():
        j = env.params.index(param)
        ref = env.X[:1].copy()
        ref[0, j] = best
        logp -= 0.5 * ((candidates[:, j] - env.encode(ref)[0, j]) / max(width, 1e-3)) ** 2
    return np.exp(logp - logp.max())


def fit_posterior(
    env: Domain,
    experiments: list[Node],
    priors: list[Prior] | None = None,
    learn: bool = True,
    seed: int = 0,
) -> Posterior:
    """Plain GP is exactly the empty-prior case.

    Gate inference is conditional on the fitted residual kernel. Signed coefficients
    have prior N(2 * trust, 1) in normalised utility units, not bounded LSTM gates.
    """
    if not experiments:
        raise ValueError("A posterior needs at least one experiment.")
    priors = priors or []
    candidates = env.encode(env.X)
    indices = [env.index(e.params) for e in experiments]
    sign = 1.0 if env.goal == "maximize" else -1.0
    y = sign * np.array([e.result for e in experiments], dtype=float)
    offset, scale = y.mean(), y.std() or 1.0
    z = (y - offset) / scale
    basis = np.array([prior_weight(env, candidates, p.belief) for p in priors]).reshape(
        len(priors), len(candidates)
    ).T
    observed = basis[indices]
    initial = 2.0 * np.array([p.trust for p in priors])
    kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(candidates.shape[1])) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=seed)
    gp.fit(candidates[indices], z - observed @ initial)
    cross = gp.kernel_(candidates[indices], candidates)
    def solve(rhs):
        return cho_solve((gp.L_, True), rhs)
    precision = np.eye(len(priors)) + observed.T @ solve(observed)
    gates = np.linalg.solve(precision, observed.T @ solve(z) + initial) if learn else initial
    mean = cross.T @ solve(z - observed @ gates) + basis @ gates
    variance = gp.kernel_.diag(candidates) - np.einsum("ij,ij->j", cross, solve(cross))
    gate_sd = np.zeros(len(priors))
    if priors and learn:
        residual = basis.T - observed.T @ solve(cross)
        variance += np.einsum("ij,ij->j", residual, np.linalg.solve(precision, residual))
        gate_sd = np.sqrt(np.diag(np.linalg.solve(precision, np.eye(len(priors)))))
    return Posterior(
        mean=sign * (mean * scale + offset),
        sd=np.sqrt(np.maximum(variance, 1e-12)) * scale,
        gates={p.id: float(g / 2) for p, g in zip(priors, gates)},
        gate_sd={p.id: float(sd / 2) for p, sd in zip(priors, gate_sd)},
    )


def expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float, goal: str, xi: float = 0.01) -> np.ndarray:
    sign = 1.0 if goal == "maximize" else -1.0
    improvement = sign * (mean - best) - xi
    z = improvement / np.maximum(sd, 1e-12)
    return improvement * norm.cdf(z) + sd * norm.pdf(z)
