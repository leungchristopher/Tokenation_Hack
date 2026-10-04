"""One surrogate and one acquisition function, plus the numerical uncertainty records it can justify."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from epistemic.tasks import TaskSpec


@dataclass
class ResponseUncertainty:
    """Statistical predictive summary. Never produced by the language model."""

    candidate_id: str
    mean: float
    latent_sd: float
    noise_sd: float
    interval: tuple[float, float]


@dataclass(frozen=True)
class NumericalPrior:
    id: str
    belief: dict[str, tuple[float, float]]
    trust: float


@dataclass
class Surrogate:
    task: TaskSpec
    seed: int = 0
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _sd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _noise: float = 0.0
    priors: list[NumericalPrior] = field(default_factory=list)
    gates: dict[str, tuple[float, float]] = field(default_factory=dict)

    def with_priors(self, priors: list[NumericalPrior]) -> "Surrogate":
        bounds = self.task.bounds()
        for prior in priors:
            if not prior.belief or not np.isfinite(prior.trust) or not 0 <= prior.trust <= 1:
                raise ValueError("A prior needs parameter beliefs and finite trust in [0, 1].")
            for name, (best, width) in prior.belief.items():
                if (name not in bounds or not np.isfinite([best, width]).all()
                        or not bounds[name][0] <= best <= bounds[name][1] or not 0 < width <= 1):
                    raise ValueError("Prior beliefs must use task parameters, bounds and positive fractional widths.")
        self.priors = priors
        return self

    def _prior_matrix(self, grid: np.ndarray) -> np.ndarray:
        rows = []
        for prior in self.priors:
            belief = np.zeros_like(grid)
            used = np.zeros(grid.shape[1], dtype=bool)
            for name, (best, width_fraction) in prior.belief.items():
                if name not in self.task.names or width_fraction <= 0:
                    continue
                column = self.task.names.index(name)
                point = np.array([self.task.X[0]], float)
                point[0, column] = best
                centre = self.task.encode(point)[0, column]
                belief[:, column] = (grid[:, column] - centre) / width_fraction
                used[column] = True
            rows.append(np.exp(-0.5 * np.square(belief[:, used]).sum(1))
                        if used.any() else np.zeros(len(grid)))
        return np.asarray(rows) if rows else np.zeros((0, len(grid)))

    def fit(self, history: list[tuple[str, float]]) -> "Surrogate":
        """Refit the residual GP and prior gates from the current observations."""
        grid = self.task.encode(self.task.X)
        if not history:
            self._mean = np.zeros(len(grid))
            self._sd = np.ones(len(grid))
            self._noise = 0.0
            self.gates = {prior.id: (prior.trust, 0.5) for prior in self.priors}
            return self
        index = {cid: i for i, cid in enumerate(self.task.ids())}
        rows = np.array([index[cid] for cid, _ in history])
        sign = 1.0 if self.task.direction == "maximize" else -1.0
        y = sign * np.array([value for _, value in history], float)
        offset, scale = float(y.mean()), float(y.std() or 1.0)
        z = (y - offset) / scale
        prior_matrix = self._prior_matrix(grid)
        prior_at_observations = prior_matrix[:, rows]
        prior_mean = 2.0 * np.array([prior.trust for prior in self.priors])
        kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(grid.shape[1])) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=self.seed, normalize_y=False)
        gp.fit(grid[rows], z - prior_at_observations.T @ prior_mean)
        if self.priors:
            fitted = gp.kernel_
            covariance = fitted(grid[rows]) + 1e-8 * np.eye(len(rows))
            cross_covariance = fitted(grid[rows], grid)
            inverse = np.linalg.inv(covariance)
            gate_precision = np.eye(len(self.priors)) + prior_at_observations @ inverse @ prior_at_observations.T
            gate_covariance = np.linalg.inv(gate_precision)
            gate_mean = gate_covariance @ (
                prior_at_observations @ inverse @ z + prior_mean
            )
            residual_prior = prior_matrix - prior_at_observations @ inverse @ cross_covariance
            mean = (
                cross_covariance.T @ inverse @ (z - prior_at_observations.T @ gate_mean)
                + prior_matrix.T @ gate_mean
            )
            variance = fitted.diag(grid) - np.einsum(
                "ij,ij->j", cross_covariance, inverse @ cross_covariance
            )
            variance += np.einsum(
                "ij,ij->j", residual_prior, gate_covariance @ residual_prior
            )
            sd = np.sqrt(np.maximum(variance, 1e-12))
            gate_sd = np.sqrt(np.diag(gate_covariance))
            self.gates = {
                prior.id: (float(mean_ / 2), float(sd_ / 2))
                for prior, mean_, sd_ in zip(self.priors, gate_mean, gate_sd)
            }
        else:
            mean, sd = gp.predict(grid, return_std=True)
            self.gates = {}
        self._mean = sign * (np.asarray(mean) * scale + offset)
        self._noise = float(np.sqrt(max(gp.kernel_.k2.noise_level, 0.0))) * scale
        self._sd = np.sqrt(np.maximum((np.asarray(sd) * scale) ** 2 - self._noise ** 2, 0.0))
        return self

    def response(self, candidate_id: str) -> ResponseUncertainty:
        i = self.task.ids().index(candidate_id)
        latent = float(self._sd[i])
        total = float(np.hypot(latent, self._noise))
        mean = float(self._mean[i])
        return ResponseUncertainty(candidate_id, mean, latent, self._noise,
                                   (mean - 1.96 * total, mean + 1.96 * total))

    def ranked(self, best: float | None, exclude: set[str] | None = None, top: int = 8) -> list[tuple[str, float]]:
        """Rank candidates by EI using the total predictive SD."""
        sign = 1.0 if self.task.direction == "maximize" else -1.0
        reference = best if best is not None else sign * -np.inf
        spread = np.hypot(self._sd, self._noise)
        if not np.isfinite(reference):
            scores = spread.copy()
        else:
            improvement = sign * (self._mean - reference) - 0.01
            z = improvement / np.maximum(spread, 1e-12)
            scores = improvement * norm.cdf(z) + spread * norm.pdf(z)
        order = np.argsort(-scores)
        ids = self.task.ids()
        chosen = [(ids[i], float(scores[i])) for i in order if ids[i] not in (exclude or set())]
        return chosen[:top]
