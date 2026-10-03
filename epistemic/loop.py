"""The one sequential loop: observe, update the model, update the epistemic state, select, execute, log."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from epistemic.evidence import seed_graph
from epistemic.execution import execution_model
from epistemic.graph import Claim, Decision, EvidenceGraph, Observation
from epistemic.policies import POLICIES, Proposal, SelectionError
from epistemic.provider import Provider, get_provider
from epistemic.surrogate import Surrogate
from epistemic.tasks import Evaluator, TaskSpec, load_task
from epistemic.update import update_state

FEEDBACK = ("true", "missing", "corrupted")


@dataclass
class Config:
    task: str = "drug"
    policy: str = "bo"
    budget: int = 12
    seed: int = 0
    execution: str = "perfect"
    feedback: str = "true"
    misleading_evidence: bool = False
    with_edges: bool = True
    with_model_advice: bool = True
    provider: str = "mock"
    corruption_scale: float = 0.5
    fault_round: int | None = None
    fault_scale: float = 1.5
    temperature: float = 0.0
    evidence_file: str | None = None
    max_searches: int = 0

    def __post_init__(self) -> None:
        if self.feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {FEEDBACK}")
        if self.budget < 0 or self.seed < 0 or self.max_searches < 0:
            raise ValueError("budget and seed must be non-negative")
        if self.fault_round is not None and not 1 <= self.fault_round <= self.budget:
            raise ValueError("fault_round must lie within the episode budget")
        if min(self.fault_scale, self.corruption_scale, self.temperature) < 0:
            raise ValueError("fault, corruption and decoding scales must be non-negative")


@dataclass
class Episode:
    config: Config
    task: TaskSpec
    graph: EvidenceGraph
    trajectory: list[dict] = field(default_factory=list)
    hidden: list[dict] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decisions(self) -> list[Decision]:
        return sorted(self.graph.decisions.values(), key=lambda d: d.round)

    def best_true(self) -> float | None:
        values = [step["true_value"] for step in self.hidden]
        if not values:
            return None
        return max(values) if self.task.direction == "maximize" else min(values)

    def final_selection(self) -> str | None:
        """What the agent would report: the executed candidate with the best value it was shown."""
        shown = [(o.value_shown, o.candidate_id) for o in self.graph.observations.values() if o.value_shown is not None]
        if not shown:
            return self.decisions[-1].candidate_id if self.decisions else None
        pick = max(shown) if self.task.direction == "maximize" else min(shown)
        return pick[1]

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
        (out / "config.json").write_text(json.dumps({**asdict(self.config), "provider": self.provider_metadata}, indent=2))
        return out


def _corrupt(task: TaskSpec, evaluator: Evaluator, candidate_id: str, value: float,
             rng: np.random.Generator, scale: float) -> float:
    """Show the agent the result of a different measured candidate; evaluator truth is unchanged."""
    other = task.ids()[int(rng.integers(len(task.ids())))]
    return float(value + scale * (evaluator.truth(other) - value))


def run_episode(config: Config, provider: Provider | None = None,
                initial: list[str] | None = None, stop_after: int | None = None,
                literature_search: Callable[[str], dict] | None = None) -> Episode:
    task, evaluator = load_task(config.task)
    provider = provider or get_provider(config.provider, seed=config.seed, temperature=config.temperature)
    policy = POLICIES[config.policy](task=task, provider=provider, with_edges=config.with_edges,
                                     with_model_advice=config.with_model_advice, budget=config.budget,
                                     max_searches=config.max_searches)
    graph = seed_graph(task, misleading=config.misleading_evidence, evidence_file=config.evidence_file)
    episode = Episode(config, task, graph, provider_metadata=provider.metadata())
    executor = execution_model(task, config.execution)
    surrogate = Surrogate(task, seed=config.seed).fit([])
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    rounds = config.budget if stop_after is None else min(config.budget, stop_after)
    searches = 0
    for round in range(1, rounds + 1):
        history = [(o.candidate_id, o.value_shown) for o in graph.observations.values() if o.value_shown is not None]
        forced = initial[round - 1] if initial and round <= len(initial) else None
        try:
            proposal = (Proposal(forced, rationale="matched initial experiment") if forced
                        else policy.propose(graph, surrogate, history, rng, round))
        except SelectionError as error:
            episode.trajectory.append({
                "round": round, "state_available_to_agent": graph.view(round, with_edges=config.with_edges),
                "prompt": getattr(policy, "last_prompt", None), "action": None,
                "llm": {"valid_output": False, "attempts": len(error.failures), "failures": error.failures},
                "status": "stopped_after_invalid_output",
            })
            break
        prediction = surrogate.response(proposal.candidate_id) if history else None
        state_view = graph.view(round, with_edges=config.with_edges)
        state_view["literature_searches_remaining"] = config.max_searches - searches
        before = {cid: len(claim.revisions) for cid, claim in graph.claims.items()}
        for index, update in enumerate(proposal.claim_updates):
            claim_id = update.claim_id or f"H{round}_{index}"
            if update.claim_id:
                graph.revise(claim_id, round, update.status, "LLM interpretation of accessible evidence",
                             update.evidence + update.contradicting_evidence, update.statement,
                             update.scope, update.discriminating_result)
            else:
                graph.add_claim(Claim(id=claim_id, statement=update.statement, scope=update.scope,
                                      source="model_conjecture", evidence=update.evidence,
                                      status=update.status, discriminating_result=update.discriminating_result), round=round)
            for ref in update.evidence:
                graph.link(ref, claim_id, "supports", "LLM interpretation; not a measured fact")
            for ref in update.contradicting_evidence:
                graph.link(ref, claim_id, "contradicts", "LLM interpretation; not causal attribution")
        decision = graph.add_decision(Decision(
            id=f"D{round}", round=round, candidate_id=proposal.candidate_id,
            claims=[e for e in proposal.evidence if e in graph.claims],
            evidence=proposal.evidence,
            assumptions=proposal.assumptions or [e for e in proposal.evidence if e in graph.assumptions],
            targeted_uncertainty=proposal.targeted_uncertainty,
            justification=proposal.rationale,
            prediction=proposal.prediction if proposal.prediction is not None else
            (prediction.mean if prediction else None),
            prediction_source="llm" if proposal.valid_llm_output else
                              ("numerical_model" if proposal.prediction is not None or prediction is not None
                               else "unavailable"),
            search_query=proposal.search_query,
            implications=proposal.implications or
                         "Outside the interval, execution, noise and model form are all candidate explanations.",
            policy=policy.name, valid_llm_output=proposal.valid_llm_output,
        ))
        for claim_id in decision.claims:
            graph.link(decision.id, claim_id, "depends_on", "Declared evidence dependency, not causal attribution.")
        for assumption_id in decision.assumptions:
            graph.link(decision.id, assumption_id, "depends_on")
        for evidence_id in decision.evidence:
            if evidence_id not in decision.claims and evidence_id not in decision.assumptions:
                graph.link(decision.id, evidence_id, "depends_on", "Accessible evidence used in this decision.")

        executor = execution_model(task, config.execution,
                                   config.fault_scale if config.fault_round == round else 0.15)
        if config.fault_round == round:
            executor = execution_model(task, "perturbed", config.fault_scale)
        record = executor.run(proposal.candidate_id, np.random.default_rng(np.random.SeedSequence([config.seed, round, 1])))
        true_value = evaluator.truth(record.realised_id)
        measured = evaluator.observe(record.realised_id, np.random.default_rng(np.random.SeedSequence([config.seed, round, 2])))
        if config.feedback == "missing":
            shown: float | None = None
        elif config.feedback == "corrupted":
            shown = _corrupt(task, evaluator, record.realised_id, measured,
                             np.random.default_rng(np.random.SeedSequence([config.seed, round, 3])), config.corruption_scale)
        else:
            shown = measured

        observation = graph.add_observation(Observation(
            id=f"O{round}", round=round, candidate_id=proposal.candidate_id,
            intended_params=record.intended_params, execution=record.accessible(),
            value_shown=shown,
            outcome_unit=task.outcome_unit, simulated=True,
        ))
        graph.link(decision.id, observation.id, "tests", "the experiment this decision ran")
        if shown is not None:
            if prediction is not None:
                surrogate.note_outcome(prediction, shown)
            surrogate.fit([(o.candidate_id, o.value_shown) for o in graph.observations.values()
                           if o.value_shown is not None])
            update_state(graph, task, observation, prediction, round)

        retrieval = None
        if proposal.search_query:
            retrieval = {"query": proposal.search_query, "status": "budget_exhausted"}
            if searches < config.max_searches and round < rounds:
                searches += 1
                policy.searches_remaining = config.max_searches - searches
                start = time.perf_counter()
                try:
                    if literature_search is None:
                        from epistemic.amass import records_to_claims, search_result
                        result = asyncio.run(search_result(proposal.search_query, limit=3))
                        claims = records_to_claims(result["records"])
                    else:
                        result = literature_search(proposal.search_query)
                        claims = result["claims"]
                    retrieved = [Claim.model_validate(payload) for payload in claims[:3]]
                    for index, claim in enumerate(retrieved):
                        if claim.source != "literature" or claim.benchmark_generated:
                            raise ValueError("Retrieval must return attributable literature, not model conjectures.")
                        claim.id = f"L{round}_{searches}_{index + 1}"
                        graph._require(*claim.evidence)
                        if claim.id in graph:
                            raise ValueError("Retrieved source ID already exists.")
                    known_sources = {
                        claim.reference: claim.id
                        for claim in graph.claims.values()
                        if claim.source == "literature" and claim.reference
                    }
                    added = []
                    duplicates = []
                    for claim in retrieved:
                        if claim.reference and claim.reference in known_sources:
                            duplicates.append({
                                "reference": claim.reference,
                                "existing_claim_id": known_sources[claim.reference],
                            })
                            continue
                        graph.add_claim(claim, round=round)
                        if claim.reference:
                            known_sources[claim.reference] = claim.id
                        added.append(claim.model_dump())
                    retrieval.update(
                        status="success",
                        claims=added,
                        duplicate_sources=duplicates,
                        tool_result=result,
                    )
                except Exception as error:
                    retrieval.update(status="failed", error_type=type(error).__name__,
                                     http_status=getattr(getattr(error, "response", None), "status_code", None))
                retrieval["latency_s"] = time.perf_counter() - start
            elif round == rounds:
                retrieval["status"] = "no_future_round"

        episode.trajectory.append({
            "round": round,
            "state_available_to_agent": state_view,
            "prompt": getattr(policy, "last_prompt", None),
            "action": decision.model_dump(),
            "epistemic_interpretations": [u.model_dump() for u in proposal.claim_updates],
            "literature_search": retrieval,
            "pre_experiment_prediction": None if prediction is None else {
                "mean": prediction.mean, "latent_sd": prediction.latent_sd,
                "noise_sd": prediction.noise_sd, "interval": list(prediction.interval)},
            "accessible_observation": {**observation.model_dump(),
                                       "value_shown": shown},
            "state_revision": {
                "new_revisions": {cid: len(c.revisions) - before.get(cid, 0)
                                  for cid, c in graph.claims.items() if len(c.revisions) != before.get(cid, 0)},
                "contradictions": len(graph.contradictions()),
                "model_uncertainty": asdict(surrogate.model_uncertainty()),
            },
            "llm": {"valid_output": proposal.valid_llm_output, "attempts": proposal.attempts,
                    "failures": proposal.failures},
        })
        episode.hidden.append({
            "round": round, "intended_id": record.intended_id, "realised_id": record.realised_id,
            "realised_params": record.realised_params, "true_value": true_value,
            "clipped_continuous_params": record.continuous_params or record.realised_params,
            "measured_value": measured, "shown_value": shown, "feedback_mode": config.feedback,
            "regret": evaluator.regret(record.realised_id),
            "seeded_execution_fault": config.fault_round == round,
            "execution_mapping_changed": record.realised_id != record.intended_id,
        })
    episode.provider_metadata = provider.metadata() | {
        "calls": provider.calls, "tokens": provider.tokens, "latency_s": round_latency(provider)}
    return episode


def round_latency(provider: Provider) -> float:
    return round(provider.latency_s, 3)
