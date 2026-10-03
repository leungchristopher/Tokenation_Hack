"""Finite candidate space and an evaluator; no benchmark truth or framework dependency."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

Parameters = dict[str, float]
Evaluator = Callable[[Parameters, np.random.Generator], float]


@dataclass
class Domain:
    name: str
    description: str
    params: list[str]
    X: np.ndarray
    evaluate: Evaluator
    goal: Literal["maximize", "minimize"] = "maximize"
    log: tuple[str, ...] = ()

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 2 or not len(self.X) or self.X.shape[1] != len(self.params):
            raise ValueError("X must be a non-empty candidate matrix with one column per parameter.")
        if not np.isfinite(self.X).all() or len(set(self.params)) != len(self.params):
            raise ValueError("Candidates must be finite and parameter names unique.")
        if self.goal not in {"maximize", "minimize"} or set(self.log) - set(self.params):
            raise ValueError("Invalid goal or unknown log-scaled parameter.")
        if np.any(self.X[:, np.isin(self.params, self.log)] <= 0):
            raise ValueError("Log-scaled candidates must be positive.")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Scale to [0, 1], using log10 for explicitly configured parameters."""
        lg = np.isin(self.params, self.log)
        X, ref = (
            np.where(lg, np.log10(np.maximum(a, 1e-12)), a)
            for a in (np.asarray(X, float), self.X)
        )
        lo, hi = ref.min(0), ref.max(0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1.0)

    def index(self, params: Parameters) -> int:
        missing = set(self.params) - set(params)
        if missing:
            raise ValueError(f"Missing parameters: {sorted(missing)}")
        x = np.array([[float(params[p]) for p in self.params]])
        if not np.isfinite(x).all() or np.any(x[:, np.isin(self.params, self.log)] <= 0):
            raise ValueError("Parameters must be finite and log-scaled values positive.")
        return int(np.argmin(((self.encode(self.X) - self.encode(x)) ** 2).sum(1)))

    def condition(self, i: int) -> Parameters:
        return dict(zip(self.params, map(float, self.X[i])))

    def sample(self, i: int, rng: np.random.Generator) -> float:
        return float(self.evaluate(self.condition(i), rng))

    def prompt(self, budget: int) -> str:
        levels = {p: sorted(np.unique(self.X[:, j]).tolist()) for j, p in enumerate(self.params)}
        ranges = "\n".join(
            f"- {p}: {v if len(v) <= 12 else f'{len(v)} levels in [{v[0]}, {v[-1]}]'}"
            for p, v in levels.items()
        )
        return (
            f"Goal: {self.goal} {self.description}.\nParameters and their available levels:\n{ranges}\n"
            f"Only {len(self.X)} combinations are feasible; requested settings are snapped to the "
            "nearest feasible condition. Measurements may be noisy.\n"
            f"Experiment budget: {budget}."
        )
