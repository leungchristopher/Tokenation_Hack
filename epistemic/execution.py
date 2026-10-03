"""Execution models. Perfect execution is the control; perturbation is seeded and clearly labelled.

Perturbation acts on physical parameter values, never on candidate-ID adjacency, which carries no
physical meaning. The realised candidate is evaluator-only; the agent sees a noisy execution report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from epistemic.tasks import TaskSpec


@dataclass
class ExecutionRecord:
    intended_id: str
    intended_params: dict[str, float]
    realised_id: str
    realised_params: dict[str, float]
    reported_params: dict[str, float]
    report_sd: dict[str, float]
    clipped: tuple[str, ...] = ()
    model: str = "perfect"
    continuous_params: dict[str, float] = field(default_factory=dict)

    def accessible(self) -> dict:
        """What the agent may see: intention, a noisy delivery report, and clipping flags."""
        return {
            "intended_params": self.intended_params,
            "reported_delivered_params": self.reported_params,
            "report_sd": self.report_sd,
            "clipped_parameters": list(self.clipped),
            "execution_model": self.model,
        }


class Executor(Protocol):
    """The only boundary a future robotics adapter would implement. No robotics import is required."""

    def run(self, candidate_id: str, rng: np.random.Generator) -> ExecutionRecord: ...


class PerfectExecution:
    """Delivers exactly what was requested."""

    name = "perfect"

    def __init__(self, task: TaskSpec) -> None:
        self.task = task

    def run(self, candidate_id: str, rng: np.random.Generator) -> ExecutionRecord:
        params = self.task.params_of(candidate_id)
        return ExecutionRecord(candidate_id, params, candidate_id, params, params,
                               {name: 0.0 for name in params}, model=self.name)


@dataclass
class PerturbedExecution:
    """Seeded relative error on delivered physical values, with explicit clipping and snapping.

    The error is simulated execution uncertainty, not measured instrument variability.
    """

    task: TaskSpec
    relative_sd: float = 0.15
    report_relative_sd: float = 0.05
    name: str = "perturbed"
    _bounds: dict[str, tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        if self.relative_sd < 0 or self.report_relative_sd < 0:
            raise ValueError("Execution perturbation scales must be non-negative.")
        self._bounds = self.task.bounds()

    def run(self, candidate_id: str, rng: np.random.Generator) -> ExecutionRecord:
        intended = self.task.params_of(candidate_id)
        realised, clipped = {}, []
        for name, value in intended.items():
            low, high = self._bounds[name]
            perturbed = (
                float(rng.normal(value, self.relative_sd * (high - low))) if name in ("ph", "temperature") or value == 0
                else value * float(np.exp(rng.normal(0.0, self.relative_sd)))
            )
            bounded = min(max(perturbed, low), high)
            if bounded != perturbed:
                clipped.append(name)
            realised[name] = float(bounded)
        realised_id = self.task.nearest(realised)
        continuous = dict(realised)
        realised = self.task.params_of(realised_id)
        reported = {n: float(v * np.exp(rng.normal(0.0, self.report_relative_sd))) for n, v in realised.items()}
        report_sd = {n: float(abs(v) * self.report_relative_sd) for n, v in reported.items()}
        return ExecutionRecord(candidate_id, intended, realised_id, realised, reported, report_sd,
                               tuple(clipped), self.name, continuous)


def execution_model(task: TaskSpec, name: str, relative_sd: float = 0.15) -> Executor:
    if name == "perfect":
        return PerfectExecution(task)
    if name == "perturbed":
        return PerturbedExecution(task, relative_sd=relative_sd)
    raise KeyError(f"Unknown execution model {name!r}.")
