"""Sequential GP-EI experimentation over finite candidate sets."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from epistemic.graph import Decision, EvidenceGraph, Observation, UncertaintyRecord
from epistemic.literature import seed_graph
from epistemic.provider import Provider, get_provider
from epistemic.surrogate import Surrogate
from epistemic.tasks import Evaluator, TaskSpec, load_task


@dataclass
class Config:
    task: str = "drug"
    budget: int = 1000
    max_seconds: float = 300.0
    max_model_tokens: int = 12_000
    patience: int = 30
    seed: int = 0
    acquisition: str = "ei"
    observation_noise: bool = True
    provider: str = "none"
    max_searches: int = 0
    noise_cv: float | None = None

    def __post_init__(self) -> None:
        if self.acquisition not in ("ei", "gated_ei"):
            raise ValueError("acquisition must be ei or gated_ei.")
        if min(self.budget, self.max_seconds, self.max_model_tokens,
               self.patience, self.seed, self.max_searches) < 0:
            raise ValueError("Budgets, seed and patience must be non-negative.")


@dataclass
class Proposal:
    candidate_id: str
    rationale: str
    claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    prior_gates: dict[str, tuple[float, float]] = field(default_factory=dict)


def select(
    task: TaskSpec, graph: EvidenceGraph, surrogate: Surrogate, rng: np.random.Generator, n_init: int = 3,
) -> Proposal:
    tried = {observation.candidate_id for observation in graph.observations.values()}
    available = [candidate_id for candidate_id in task.ids() if candidate_id not in tried] or task.ids()
    assumptions: list[str] = [
        record_id for record_id in ("A_gp", "A0") if record_id in graph.assumptions
    ]
    if len(graph.observations) < n_init:
        return Proposal(
            available[int(rng.integers(len(available)))],
            "Initial measured-candidate design.",
            assumptions=assumptions,
        )
    shown = [
        observation.value_shown for observation in graph.observations.values()
        if observation.value_shown is not None
    ]
    best = (max if task.direction == "maximize" else min)(shown) if shown else None
    ranked = surrogate.ranked(best, exclude=tried, top=1) or surrogate.ranked(best, top=1)
    candidate_id, score = ranked[0]
    gates = ", ".join(
        f"{prior_id} {mean:.2f}±{sd:.2f}" for prior_id, (mean, sd) in surrogate.gates.items()
    )
    rationale = f"Highest expected improvement ({score:.5g}) under the fitted GP; not proof of optimality."
    if gates:
        rationale += f" Learned literature gates: {gates}."
    claims = [prior.id for prior in surrogate.priors if prior.id in graph.claims]
    evidence = list(dict.fromkeys(
        evidence_id for claim_id in claims for evidence_id in graph.claims[claim_id].evidence
    ))
    return Proposal(
        candidate_id, rationale, claims=claims, evidence=evidence, assumptions=assumptions,
        prior_gates=dict(surrogate.gates),
    )


@dataclass
class Episode:
    config: Config
    task: TaskSpec
    graph: EvidenceGraph
    trajectory: list[dict] = field(default_factory=list)
    hidden: list[dict] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    evaluator_provenance: str = ""
    evaluator: Evaluator | None = field(default=None, repr=False)
    stop_reason: str = "not_started"
    elapsed_s: float = 0.0
    initial_evidence: dict[str, Any] = field(default_factory=dict)

    def final_selection(self) -> str | None:
        result = self.final_result()
        return result["candidate_id"] if result else None

    def final_result(self) -> dict[str, Any] | None:
        shown = sorted(
            (observation for observation in self.graph.observations.values()
             if observation.value_shown is not None),
            key=lambda observation: observation.round,
        )
        if not shown:
            return None
        pick = (max if self.task.direction == "maximize" else min)(
            shown, key=lambda observation: (
                observation.value_shown if observation.value_shown is not None else 0.0
            ),
        )
        return {
            "candidate_id": pick.candidate_id,
            "observation_id": pick.id,
            "value_shown": pick.value_shown,
            "outcome_unit": self.task.outcome_unit,
            "rule": f"Greedy {self.task.direction} over observed results; ties keep the earliest experiment.",
            "uncertainty": (
                "Best observed, not a proven optimum. Observation noise can misrank candidates; "
                "predictions do not determine final selection."
            ),
            "uncertainty_records": [
                record.id for record in self.graph.uncertainties.values() if pick.id in record.evidence
            ],
        }

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "trajectory.jsonl").open("x") as handle:
            for step in self.trajectory:
                handle.write(json.dumps(step) + "\n")
        with (out / "hidden_truth.jsonl").open("x") as handle:
            for step in self.hidden:
                handle.write(json.dumps(step) + "\n")
        (out / "graph.json").write_text(self.graph.model_dump_json(indent=2))
        (out / "final.json").write_text(json.dumps(self.final_result(), indent=2))
        (out / "config.json").write_text(json.dumps(
            {**asdict(self.config), "provider": self.provider_metadata}, indent=2,
        ))
        (out / "termination.json").write_text(json.dumps({
            "reason": self.stop_reason,
            "elapsed_s": self.elapsed_s,
            "experiments": len(self.hidden),
            "model_tokens": self.provider_metadata.get("tokens", 0),
        }, indent=2))
        (out / "literature_setup.json").write_text(json.dumps(self.initial_evidence, indent=2))
        (out / "hidden_provenance.json").write_text(json.dumps({"source": self.evaluator_provenance}, indent=2))
        return out


def run_episode(
    config: Config,
    provider: Provider | None = None,
    literature_search: Callable[[str], dict] | None = None,
    domain: tuple[TaskSpec, Evaluator] | None = None,
) -> Episode:
    task, evaluator = domain or load_task(config.task, noise_cv=config.noise_cv)
    provider = provider or get_provider(config.provider)
    started = time.perf_counter()
    provider.deadline = started + config.max_seconds
    provider.token_limit = config.max_model_tokens
    graph = seed_graph(task)
    priors = []
    setup_limit = None
    literature_setup: dict[str, Any] = {}
    if config.acquisition == "gated_ei":
        from epistemic.literature import initialise_literature_priors

        setup = initialise_literature_priors(
            task, graph, provider, config.max_searches, literature_search, experiment_cap=config.budget,
        )
        priors = setup.priors
        setup_limit = setup.limit_reason
        literature_setup = {
            "priors": [asdict(prior) for prior in setup.priors],
            "searches": setup.searches,
            "failures": setup.failures,
            "limit_reason": setup.limit_reason,
        }
    episode = Episode(
        config, task, graph, provider_metadata=provider.metadata(),
        evaluator_provenance=evaluator.provenance, evaluator=evaluator, initial_evidence=literature_setup,
    )
    if setup_limit:
        episode.stop_reason = setup_limit
    surrogate = Surrogate(task, seed=config.seed).with_priors(priors).fit([])
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    best_observed: float | None = None
    last_improvement: int | None = None

    for round_index in range(1, config.budget + 1):
        if episode.stop_reason != "not_started":
            break
        if time.perf_counter() >= provider.deadline:
            episode.stop_reason = "time_limit"
            break
        if provider.name != "none" and provider.tokens >= config.max_model_tokens:
            episode.stop_reason = "model_token_limit"
            break
        if (config.patience and last_improvement is not None
                and round_index - 1 - last_improvement >= config.patience):
            episode.stop_reason = "stagnation"
            break

        proposal = select(task, graph, surrogate, rng)
        history = [
            (observation.candidate_id, observation.value_shown)
            for observation in graph.observations.values()
            if observation.value_shown is not None
        ]
        prediction = surrogate.response(proposal.candidate_id) if history else None
        decision = graph.add_decision(Decision(
            id=f"D{round_index}",
            round=round_index,
            candidate_id=proposal.candidate_id,
            claims=proposal.claims,
            evidence=proposal.evidence,
            assumptions=proposal.assumptions,
            justification=proposal.rationale,
            prediction=prediction.mean if prediction else None,
            prediction_source="numerical_model" if prediction else "unavailable",
            prior_gates=proposal.prior_gates,
        ))
        for claim_id in decision.claims:
            gate = surrogate.gates.get(claim_id)
            initial_trust = graph.claims[claim_id].trust
            note = (
                f"gate {gate[0]:.2f} ± {gate[1]:.2f}; initial trust {initial_trust:.2f}"
                if gate and initial_trust is not None else "Literature prior used by the GP."
            )
            graph.link(decision.id, claim_id, "depends_on", note)
        for record_id in [*decision.evidence, *decision.assumptions]:
            graph.link(decision.id, record_id, "depends_on")

        candidate_id = proposal.candidate_id
        true_value = evaluator.truth(candidate_id)
        shown = (
            evaluator.observe(candidate_id, np.random.default_rng(
                np.random.SeedSequence([config.seed, round_index, 2]),
            ))
            if config.observation_noise else true_value
        )
        observation = graph.add_observation(Observation(
            id=f"O{round_index}",
            round=round_index,
            candidate_id=candidate_id,
            intended_params=task.params_of(candidate_id),
            value_shown=shown,
            outcome_unit=task.outcome_unit,
            simulated=True,
            measurement_noise=config.observation_noise,
        ))
        graph.link(decision.id, observation.id, "tests", "the experiment this decision ran")
        uncertainty = graph.add_uncertainty(UncertaintyRecord(
            id=f"U_O{round_index}",
            category="response",
            statement=(task.noise.description if config.observation_noise
                       else "Measurement noise was disabled for this observation."),
            scope=task.name,
            evidence=(observation.id,),
            round=round_index,
        ))
        graph.link(uncertainty.id, observation.id, "qualifies")

        improved = (
            best_observed is None
            or (shown > best_observed if task.direction == "maximize" else shown < best_observed)
        )
        if improved:
            best_observed = shown
            last_improvement = round_index
        surrogate.fit([
            (item.candidate_id, item.value_shown) for item in graph.observations.values()
            if item.value_shown is not None
        ])
        episode.trajectory.append({
            "round": round_index,
            "action": decision.model_dump(mode="json"),
            "pre_experiment_prediction": None if prediction is None else {
                "mean": prediction.mean,
                "latent_sd": prediction.latent_sd,
                "noise_sd": prediction.noise_sd,
                "interval": list(prediction.interval),
            },
            "accessible_observation": observation.model_dump(mode="json"),
        })
        episode.hidden.append({
            "round": round_index,
            "candidate_id": candidate_id,
            "true_value": true_value,
            "measured_value": shown,
            "regret": evaluator.regret(candidate_id),
        })
    else:
        episode.stop_reason = "experiment_limit"

    episode.elapsed_s = round(time.perf_counter() - started, 3)
    episode.provider_metadata = provider.metadata() | {
        "calls": provider.calls,
        "tokens": provider.tokens,
        "latency_s": round(provider.latency_s, 3),
    }
    return episode
