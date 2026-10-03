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
    source: str = "gaussian_process"


@dataclass
class ModelUncertainty:
    """Assumptions, fit diagnostics and alternative explanations for misfit."""

    assumptions: tuple[str, ...] = (
        "a smooth Matern-5/2 response over scaled parameters",
        "additive, roughly constant-variance observation noise",
        "the measured candidate set is representative of the reachable space",
    )
    observations: int = 0
    residual_rmse: float | None = None
    interval_coverage: float | None = None
    candidate_explanations: tuple[str, ...] = ()
    note: str = ("Large residuals are a prompt to check execution, noise and model form; on their own they "
                 "do not establish a mechanism or a model failure.")


@dataclass
class Surrogate:
    task: TaskSpec
    seed: int = 0
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _sd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _noise: float = 0.0
    residuals: list[float] = field(default_factory=list)
    covered: list[bool] = field(default_factory=list)
    observations: int = 0

    def fit(self, history: list[tuple[str, float]]) -> "Surrogate":
        """Refit from scratch, so derived state can never survive an intervention on history."""
        self.observations = len(history)
        grid = self.task.encode(self.task.X)
        if not history:
            self._mean = np.zeros(len(grid))
            self._sd = np.ones(len(grid))
            self._noise = 0.0
            return self
        index = {cid: i for i, cid in enumerate(self.task.ids())}
        rows = np.array([index[cid] for cid, _ in history])
        y = np.array([value for _, value in history], float)
        offset, scale = float(y.mean()), float(y.std() or 1.0)
        kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(grid.shape[1])) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=self.seed, normalize_y=False)
        gp.fit(grid[rows], (y - offset) / scale)
        mean, sd = gp.predict(grid, return_std=True)
        self._mean = mean * scale + offset
        self._noise = float(np.sqrt(max(gp.kernel_.k2.noise_level, 0.0))) * scale
        self._sd = np.sqrt(np.maximum((np.asarray(sd) * scale) ** 2 - self._noise ** 2, 0.0))
        return self

    def note_outcome(self, prediction: ResponseUncertainty, observed: float) -> None:
        """Score the prediction made before the experiment against what came back."""
        self.residuals.append(float(observed - prediction.mean))
        self.covered.append(bool(prediction.interval[0] <= observed <= prediction.interval[1]))

    def response(self, candidate_id: str) -> ResponseUncertainty:
        i = self.task.ids().index(candidate_id)
        latent = float(self._sd[i])
        total = float(np.hypot(latent, self._noise))
        mean = float(self._mean[i])
        return ResponseUncertainty(candidate_id, mean, latent, self._noise,
                                   (mean - 1.96 * total, mean + 1.96 * total))

    def model_uncertainty(self) -> ModelUncertainty:
        rmse = float(np.sqrt(np.mean(np.square(self.residuals)))) if self.residuals else None
        coverage = float(np.mean(self.covered)) if self.covered else None
        explanations = () if rmse is None else (
            "simulated observation noise",
            "execution differing from the intended condition",
            "a response sharper than the fitted smoothness",
        )
        return ModelUncertainty(observations=self.observations, residual_rmse=rmse,
                                interval_coverage=coverage, candidate_explanations=explanations)

    def ranked(self, best: float | None, exclude: set[str] | None = None, top: int = 8) -> list[tuple[str, float]]:
        """Expected improvement over all candidates, highest first."""
        sign = 1.0 if self.task.direction == "maximize" else -1.0
        reference = best if best is not None else sign * -np.inf
        if not np.isfinite(reference):
            scores = self._sd.copy()
        else:
            improvement = sign * (self._mean - reference) - 0.01
            z = improvement / np.maximum(self._sd, 1e-12)
            scores = improvement * norm.cdf(z) + self._sd * norm.pdf(z)
        order = np.argsort(-scores)
        ids = self.task.ids()
        chosen = [(ids[i], float(scores[i])) for i in order if ids[i] not in (exclude or set())]
        return chosen[:top]
