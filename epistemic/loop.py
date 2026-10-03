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

from epistemic.evidence import ingest_literature, seed_graph
from epistemic.execution import execution_model
from epistemic.graph import Claim, Decision, EvidenceGraph, Observation, UncertaintyKind, UncertaintyRecord
from epistemic.policies import EvidenceAnnotator, Proposal, select
from epistemic.provider import Provider, get_provider
from epistemic.source_filter import accessible_result, excluded_source
from epistemic.surrogate import Surrogate
from epistemic.tasks import Evaluator, TaskSpec, load_task
from epistemic.update import update_state


@dataclass
class Config:
    task: str = "drug"
    budget: int = 12
    seed: int = 0
    execution: str = "perfect"
    acquisition: str = "ei"
    observation_noise: bool = True
    with_evidence: bool = True
    with_edges: bool = True
    provider: str = "none"
    temperature: float = 0.0
    evidence_file: str | None = None
    max_searches: int = 0
    max_evidence_calls: int = 3

    def __post_init__(self) -> None:
        if self.acquisition not in ("ei", "random") or self.execution not in ("perfect", "perturbed"):
            raise ValueError("Use ei/random acquisition and perfect/perturbed execution.")
        if min(self.budget, self.seed, self.max_searches, self.max_evidence_calls, self.temperature) < 0:
            raise ValueError("Budgets, seed and temperature must be non-negative.")


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

    @property
    def decisions(self) -> list[Decision]:
        return sorted(self.graph.decisions.values(), key=lambda d: d.round)

    def best_true(self) -> float | None:
        values = [step["true_value"] for step in self.hidden]
        if not values:
            return None
        return max(values) if self.task.direction == "maximize" else min(values)

    def final_selection(self) -> str | None:
        result = self.final_result()
        return result["candidate_id"] if result else None

    def final_result(self) -> dict[str, Any] | None:
        shown = [o for o in self.graph.observations.values() if o.value_shown is not None]
        if not shown:
            return None
        pick = (max if self.task.direction == "maximize" else min)(shown, key=lambda o: float(o.value_shown or 0.0))
        return {
            "candidate_id": pick.candidate_id,
            "observation_id": pick.id,
            "value_shown": pick.value_shown,
            "outcome_unit": self.task.outcome_unit,
            "rule": f"Greedy {self.task.direction} over observed results; ties keep the earliest experiment.",
            "uncertainty": (
                "Best observed, not a proven optimum. Observation noise can misrank candidates; "
                "execution may differ from intended settings. No hidden dataset means or GP forecasts "
                "are used for final selection."
            ),
            "uncertainty_records": [
                u.id for u in self.graph.uncertainties.values() if pick.id in u.evidence
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
        (out / "config.json").write_text(json.dumps({**asdict(self.config), "provider": self.provider_metadata}, indent=2))
        (out / "hidden_provenance.json").write_text(json.dumps({"source": self.evaluator_provenance}, indent=2))
        return out


def run_episode(config: Config, provider: Provider | None = None,
                initial: list[str] | None = None, literature_search: Callable[[str], dict] | None = None,
                domain: tuple[TaskSpec, Evaluator] | None = None) -> Episode:
    task, evaluator = domain or load_task(config.task)
    provider = provider or get_provider(config.provider, seed=config.seed, temperature=config.temperature)
    annotator = EvidenceAnnotator(task, provider, config.budget, config.max_searches,
                                 config.max_evidence_calls, config.with_edges)
    graph = seed_graph(task, evidence_file=config.evidence_file if config.with_evidence else None)
    episode = Episode(config, task, graph, provider_metadata=provider.metadata(),
                      evaluator_provenance=evaluator.provenance, evaluator=evaluator)
    executor = execution_model(task, config.execution)
    surrogate = Surrogate(task, seed=config.seed).fit([])
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    rounds = config.budget
    searches = 0
    for round in range(1, rounds + 1):
        history = [(o.candidate_id, o.value_shown) for o in graph.observations.values() if o.value_shown is not None]
        forced = initial[round - 1] if initial and round <= len(initial) else None
        proposal = (Proposal(forced, rationale="Matched initial experiment.") if forced
                    else select(task, graph, surrogate, rng, config.acquisition))
        if config.with_evidence and provider.name != "none":
            annotator.annotate(proposal, graph, surrogate, round)
        prediction = surrogate.response(proposal.candidate_id) if history else None
        state_view = graph.view(round, with_edges=config.with_edges)
        state_view["literature_searches_remaining"] = config.max_searches - searches
        before = {cid: len(claim.revisions) for cid, claim in graph.claims.items()}
        for index, update in enumerate(proposal.claim_updates):
            claim_id = update.claim_id or f"H{round}_{index}"
            if update.claim_id:
                graph.revise(claim_id, round, update.status, "LLM interpretation of accessible evidence",
                             update.evidence + update.contradicting_evidence + update.qualifying_evidence
                             + update.non_transferable_evidence, update.statement,
                             update.scope, update.discriminating_result)
            else:
                uncertainty_ids = []
                categories: tuple[UncertaintyKind, ...] = ("source", "transfer", "mechanistic")
                for category in categories:
                    item = graph.add_uncertainty(UncertaintyRecord(
                        id=f"U_{claim_id}_{category}", category=category,
                        statement=f"{category.capitalize()} uncertainty has not been resolved by this interpretation.",
                        scope=update.scope, evidence=tuple(update.evidence), round=round,
                    ))
                    uncertainty_ids.append(item.id)
                graph.add_claim(Claim(id=claim_id, statement=update.statement, scope=update.scope,
                                      source="model_conjecture", evidence=update.evidence,
                                      status=update.status, uncertainties=uncertainty_ids,
                                      discriminating_result=update.discriminating_result), round=round)
                for uncertainty_id in uncertainty_ids:
                    graph.link(uncertainty_id, claim_id, "qualifies")
            for ref in update.evidence:
                graph.link(ref, claim_id, "supports", "LLM interpretation; not a measured fact")
            for ref in update.contradicting_evidence:
                graph.link(ref, claim_id, "contradicts", "LLM interpretation; not causal attribution")
            for ref in update.qualifying_evidence:
                graph.link(ref, claim_id, "qualifies", "Interpretation narrows, but does not refute, the claim.")
            for ref in update.non_transferable_evidence:
                graph.link(ref, claim_id, "not_transferable", "Interpretation rejects the proposed context transfer.")
        decision = graph.add_decision(Decision(
            id=f"D{round}", round=round, candidate_id=proposal.candidate_id,
            claims=[e for e in proposal.evidence if e in graph.claims],
            evidence=proposal.evidence,
            assumptions=proposal.assumptions or [e for e in proposal.evidence if e in graph.assumptions],
            targeted_uncertainty="response",
            justification=proposal.rationale,
            prediction=proposal.prediction if proposal.prediction is not None else
            (prediction.mean if prediction else None),
            prediction_source="numerical_model" if prediction is not None else "unavailable",
            search_query=proposal.search_query,
            implications="Outside the interval, execution, noise and model form are all candidate explanations.",
            policy="gp_bo",
        ))
        for claim_id in decision.claims:
            graph.link(decision.id, claim_id, "depends_on", "Declared evidence dependency, not causal attribution.")
        for assumption_id in decision.assumptions:
            graph.link(decision.id, assumption_id, "depends_on")
        for evidence_id in decision.evidence:
            if evidence_id not in decision.claims and evidence_id not in decision.assumptions:
                graph.link(decision.id, evidence_id, "depends_on", "Accessible evidence used in this decision.")

        record = executor.run(proposal.candidate_id, np.random.default_rng(np.random.SeedSequence([config.seed, round, 1])))
        true_value = evaluator.truth(record.realised_id)
        shown = (evaluator.observe(record.realised_id, np.random.default_rng(np.random.SeedSequence([config.seed, round, 2])))
                 if config.observation_noise else true_value)

        observation = graph.add_observation(Observation(
            id=f"O{round}", round=round, candidate_id=proposal.candidate_id,
            intended_params=record.intended_params, execution=record.accessible(),
            value_shown=shown,
            outcome_unit=task.outcome_unit, simulated=True, measurement_noise=config.observation_noise,
        ))
        graph.link(decision.id, observation.id, "tests", "the experiment this decision ran")
        if shown is not None:
            if prediction is not None:
                surrogate.note_outcome(prediction, shown)
            surrogate.fit([(o.candidate_id, o.value_shown) for o in graph.observations.values()
                           if o.value_shown is not None])
            update_state(graph, task, observation, prediction, round)

        retrieval: dict[str, Any] | None = None
        if proposal.search_query:
            retrieval = {"query": proposal.search_query, "status": "budget_exhausted"}
            if searches < config.max_searches and round < rounds:
                searches += 1
                annotator.searches_remaining = config.max_searches - searches
                start = time.perf_counter()
                try:
                    if excluded_source(proposal.search_query):
                        raise ValueError("This source is excluded from evaluation evidence.")
                    if literature_search is None:
                        from epistemic.amass import records_to_claims, search_result
                        result = asyncio.run(search_result(proposal.search_query, limit=3))
                        bundle = records_to_claims(result["records"], query=proposal.search_query)
                    else:
                        result = accessible_result(literature_search(proposal.search_query))
                        bundle = result
                    added, duplicates = ingest_literature(
                        graph, bundle, round=round, prefix=f"{round}_{searches}", query=proposal.search_query, limit=3,
                    )
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
            "prompt": annotator.last_prompt,
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
            "evidence_annotation": {"valid": proposal.evidence_valid, "failures": proposal.failures},
        })
        episode.hidden.append({
            "round": round, "intended_id": record.intended_id, "realised_id": record.realised_id,
            "realised_params": record.realised_params, "true_value": true_value,
            "clipped_continuous_params": record.continuous_params or record.realised_params,
            "measured_value": shown,
            "regret": evaluator.regret(record.realised_id),
            "execution_mapping_changed": record.realised_id != record.intended_id,
        })
    episode.provider_metadata = provider.metadata() | {
        "calls": provider.calls, "tokens": provider.tokens, "latency_s": round_latency(provider)}
    return episode


def round_latency(provider: Provider) -> float:
    return round(provider.latency_s, 3)
