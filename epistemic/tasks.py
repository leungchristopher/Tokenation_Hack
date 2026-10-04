"""The two measured tasks. Outcomes stay evaluator-only; agents see candidates and executed results."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DRUG_NOISE_CV = 0.05

DATA = Path(__file__).resolve().parent.parent / "data"
if not DATA.is_dir():
    DATA = Path(sys.prefix) / "share" / "bo-eval" / "data"


@dataclass(frozen=True)
class Parameter:
    name: str
    unit: str
    log: bool = False


@dataclass(frozen=True)
class NoiseSpec:
    """Where the per-candidate spread comes from. Simulated draws are never called measured replicates."""

    kind: str
    description: str
    replicates: bool


@dataclass
class TaskSpec:
    name: str
    description: str
    parameters: tuple[Parameter, ...]
    candidates: pd.DataFrame
    objective: str
    direction: str
    outcome_unit: str
    provenance: str
    limitations: tuple[str, ...]
    noise: NoiseSpec

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if self.candidates["candidate_id"].duplicated().any():
            raise ValueError("Candidate IDs must be unique.")
        if self.candidates[self.names].duplicated().any():
            raise ValueError("Measured candidate parameters must be distinct.")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def X(self) -> np.ndarray:
        return self.candidates[self.names].to_numpy(float)

    def ids(self) -> list[str]:
        return self.candidates["candidate_id"].tolist()

    def params_of(self, candidate_id: str) -> dict[str, float]:
        row = self.candidates.loc[self.candidates["candidate_id"] == candidate_id]
        if row.empty:
            raise KeyError(f"Unknown candidate_id {candidate_id!r}.")
        return {name: float(row.iloc[0][name]) for name in self.names}

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Scale to [0, 1]; log10 for parameters declared logarithmic (doses span three decades)."""
        logs = np.array([p.log for p in self.parameters])
        values, reference = (
            np.where(logs, np.log10(np.maximum(np.asarray(a, float), 1e-12)), np.asarray(a, float))
            for a in (np.atleast_2d(X), self.X)
        )
        lo, hi = reference.min(0), reference.max(0)
        return (values - lo) / np.where(hi > lo, hi - lo, 1.0)

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {name: (float(self.candidates[name].min()), float(self.candidates[name].max())) for name in self.names}

    def briefing(self, budget: int) -> str:
        levels = {
            p.name: sorted(self.candidates[p.name].unique().tolist()) for p in self.parameters
        }
        described = "\n".join(
            f"- {p.name} [{p.unit}]"
            + (f": {levels[p.name]}" if len(levels[p.name]) <= 10
               else f": {len(levels[p.name])} measured levels in [{levels[p.name][0]}, {levels[p.name][-1]}]")
            for p in self.parameters
        )
        return (
            f"Task: {self.description}\n"
            f"Objective: {self.direction} {self.objective} [{self.outcome_unit}].\n"
            f"Parameters:\n{described}\n"
            f"{len(self.candidates)} measured candidates are selectable by candidate_id.\n"
            f"Observation noise: {self.noise.description}\n"
            f"Budget: {budget} experiments (repeats allowed and charged)."
        )


class Evaluator:
    """Holds hidden truth. Agents never receive unqueried labels."""

    def __init__(self, task: TaskSpec, outcomes: pd.DataFrame, provenance: str | None = None) -> None:
        self._task = task
        self.provenance = provenance or task.provenance
        if outcomes["candidate_id"].tolist() != task.ids():
            raise ValueError("Evaluator outcomes must match the public candidate IDs.")
        self._outcomes = outcomes.set_index("candidate_id")
        values = self._outcomes["outcome"].to_numpy(float)
        self.optimum_id = str(self._outcomes.index[int(np.argmin(values) if task.direction == "minimize" else np.argmax(values))])
        self.optimum_value = float(self._outcomes.loc[self.optimum_id, "outcome"])

    def truth(self, candidate_id: str) -> float:
        return float(self._outcomes.loc[candidate_id, "outcome"])

    def spread(self, candidate_id: str) -> float:
        return float(self._outcomes.loc[candidate_id, "spread"])

    def observe(self, candidate_id: str, rng: np.random.Generator) -> float:
        """One simulated measurement around the recorded outcome."""
        value = rng.normal(self.truth(candidate_id), self.spread(candidate_id))
        return float(max(value, 0.0))

    def regret(self, candidate_id: str) -> float:
        gap = abs(self.truth(candidate_id) - self.optimum_value)
        return gap / abs(self.optimum_value) if self.optimum_value else gap


def _assemble(frame: pd.DataFrame, names: list[str], outcome: str, spread: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values(names).reset_index(drop=True)
    ids = [f"C{i:04d}" for i in range(len(frame))]
    candidates = frame[names].copy()
    candidates.insert(0, "candidate_id", ids)
    outcomes = pd.DataFrame({
        "candidate_id": ids,
        "outcome": frame[outcome].to_numpy(float),
        "spread": frame[spread].to_numpy(float),
    })
    return candidates, outcomes


def load_enzyme() -> tuple[TaskSpec, Evaluator]:
    frame = pd.read_csv(DATA / "upo_abts.csv")
    names = ["ph", "salt_conc", "cosubstrate_conc", "organic_solvent_conc", "temperature"]
    pooled = frame.assign(_var=frame["rate_sd"] ** 2).groupby(names, as_index=False).agg(
        rate_mean=("rate_mean", "mean"), _var=("_var", "mean"), rows=("rate_mean", "size")
    )
    pooled["rate_sd"] = np.sqrt(pooled.pop("_var"))
    candidates, outcomes = _assemble(pooled, names, "rate_mean", "rate_sd")
    task = TaskSpec(
        name="enzyme_upo_abts",
        description="unspecific peroxygenase (UPO) oxidising ABTS under varied assay conditions",
        parameters=(
            Parameter("ph", "pH units"),
            Parameter("salt_conc", "unit not recorded in the dataset"),
            Parameter("cosubstrate_conc", "unit not recorded in the dataset"),
            Parameter("organic_solvent_conc", "unit not recorded in the dataset"),
            Parameter("temperature", "unit not recorded in the dataset"),
        ),
        candidates=candidates,
        objective="mean specific rate",
        direction="maximize",
        outcome_unit="U/mg according to the existing adapter; upstream units unverified",
        provenance="data/upo_abts.csv, committed with the original UPO-ABTS harness in this repository.",
        limitations=(
            "The 814 measured conditions are a sparse, unbalanced subset of the 11,760-point level grid, "
            "so selection is restricted to measured candidates and no interpolation environment is claimed.",
            "Units for salt, cosubstrate and organic solvent concentration are not recorded in the CSV.",
            "Upstream collection protocol and replicate design are not documented in this repository.",
            "The original 818 rows include four repeated parameter conditions; recorded means are averaged "
            "and reported squared spreads are averaged as in the original adapter. Repeated summary rows "
            "are not treated as raw replicates.",
        ),
        noise=NoiseSpec(
            kind="dataset_reported_spread",
            description="Each draw is simulated from the dataset's reported per-condition standard deviation; "
                        "the underlying replicate design is undocumented here.",
            replicates=False,
        ),
    )
    return task, Evaluator(task, outcomes)


def load_drug(noise_cv: float = DRUG_NOISE_CV) -> tuple[TaskSpec, Evaluator]:
    if not np.isfinite(noise_cv) or noise_cv < 0:
        raise ValueError("noise_cv must be non-negative.")
    frame = pd.read_csv(DATA / "zimmer/a549_taxol_cis_dox.csv")
    names = ["taxol_uM", "cisplatin_uM", "doxorubicin_uM"]
    if len(frame) != 512 or frame[names].duplicated().any():
        raise ValueError("Expected 512 distinct three-drug combinations.")
    candidates, outcomes = _assemble(frame, names, "survival_mean", "survival_sd")
    outcomes["spread"] = noise_cv * outcomes["outcome"]
    task = TaskSpec(
        name="drug_a549_taxol_cis_dox",
        description="A549 lung cancer cell survival after 48 h of simultaneous taxol, cisplatin and doxorubicin",
        parameters=(
            Parameter("taxol_uM", "micromolar", log=True),
            Parameter("cisplatin_uM", "micromolar", log=True),
            Parameter("doxorubicin_uM", "micromolar", log=True),
        ),
        candidates=candidates,
        objective="percentage of surviving cells",
        direction="minimize",
        outcome_unit="% survival",
        provenance="benchmark:drug-response; source attribution is withheld from the agent for evaluation.",
        limitations=(
            "The table holds one value per combination with no replicates; noise uses a configurable CV, not an empirical estimate.",
            "Single-agent and vehicle controls are not included here, so no synergy objective is defined and "
            "low combination survival must not be reported as synergy.",
            "Only the three-drug column is used; the dataset's lower-order combination columns are not loaded.",
        ),
        noise=NoiseSpec(
            kind="relative_cv",
            description=f"Simulated measurement noise with standard deviation {noise_cv:.0%} of the recorded survival. "
                        "The dataset has no replicates; this CV is a modelling choice, not an empirical estimate.",
            replicates=False,
        ),
    )
    return task, Evaluator(
        task, outcomes,
        provenance="Zimmer et al., PNAS 2016 (doi:10.1073/pnas.1606301113), via the S2 archive of Tendler et al., "
                   "PLoS Comput Biol 2019 (doi:10.1371/journal.pcbi.1006956.s002); converted by data/zimmer/convert.py.",
    )


TASKS = {"enzyme": load_enzyme, "drug": load_drug}


def load_task(name: str, noise_cv: float | None = None) -> tuple[TaskSpec, Evaluator]:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    if noise_cv is not None:
        if name != "drug":
            raise ValueError("noise_cv can only be configured for the drug task.")
        return load_drug(noise_cv)
    return TASKS[name]()
