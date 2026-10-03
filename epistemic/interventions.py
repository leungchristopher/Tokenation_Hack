"""Feedback interventions and graph controls. All derived state is rebuilt in every arm."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from epistemic.evidence import seed_graph
from epistemic.graph import Claim, EvidenceGraph, Observation
from epistemic.loop import Config, run_episode
from epistemic.metrics import episode_metrics, recovery_round
from epistemic.policies import POLICIES, Proposal, SelectionError
from epistemic.provider import get_provider
from epistemic.surrogate import Surrogate
from epistemic.tasks import TaskSpec, load_task
from epistemic.update import update_state

ARMS = ("true", "removed", "permuted", "contradicted")


def intervene(observations: list[Observation], task: TaskSpec, arm: str,
              seed: int = 0, target_id: str | None = None) -> list[Observation]:
    """Modify outcomes only, keeping intended settings and accessible execution observations fixed."""
    if arm not in ARMS:
        raise KeyError(arm)
    copies = [o.model_copy(deep=True) for o in observations]
    if arm == "true":
        return copies
    if arm == "removed":
        return [o.model_copy(update={"value_shown": None}) for o in copies]
    positions = [i for i, o in enumerate(copies) if o.value_shown is not None]
    values = [float(copies[i].value_shown) for i in positions]  # type: ignore[arg-type]
    if not values:
        return copies
    if arm == "permuted":
        values = np.random.default_rng(seed).permutation(values).tolist()
        for i, value in zip(positions, values):
            copies[i] = copies[i].model_copy(update={"value_shown": value})
    else:
        if target_id:
            index = next(i for i, o in enumerate(copies) if o.id == target_id)
        else:
            best = np.argmin(values) if task.direction == "minimize" else np.argmax(values)
            index = positions[int(best)]
        spread = float(np.std(values) or abs(np.mean(values)) or 1.0)
        # Worsen the best prior result to test its influence; these are counterfactual inputs, not measurements.
        new_value = max(values) + 3 * spread if task.direction == "minimize" else min(values) - 3 * spread
        copies[index] = copies[index].model_copy(update={"value_shown": new_value})
    return copies


def rebuild(task: TaskSpec, observations: list[Observation], seed: int,
            misleading: bool = False, model_advice: bool = True,
            evidence_file: str | None = None,
            frozen_evidence: list[Claim] | None = None) -> tuple[EvidenceGraph, Surrogate, list[tuple[str, float]]]:
    """Replay observations, not stored claims or interpretations of true feedback."""
    graph = seed_graph(task, misleading, evidence_file)
    for claim in frozen_evidence or []:
        if claim.source != "literature":
            raise ValueError("Only external literature, not feedback-derived claims, may be frozen.")
        if claim.id not in graph:
            graph.add_claim(claim.model_copy(deep=True))
    surrogate = Surrogate(task, seed)
    history: list[tuple[str, float]] = []
    for observation in observations:
        surrogate.fit(history)
        prediction = surrogate.response(observation.candidate_id) if history and model_advice else None
        graph.add_observation(observation)
        if observation.value_shown is not None:
            if prediction:
                surrogate.note_outcome(prediction, observation.value_shown)
            update_state(graph, task, observation, prediction, observation.round)
            history.append((observation.candidate_id, observation.value_shown))
    surrogate.fit(history)
    return graph, surrogate, history


def next_decision(config: Config, observations: list[Observation], arm: str, sampling_seed: int,
                  intervention_seed: int = 0, target_id: str | None = None,
                  frozen_evidence: list[Claim] | None = None) -> tuple[Proposal, dict]:
    task, _ = load_task(config.task)
    modified = intervene(observations, task, arm, intervention_seed, target_id)
    graph, surrogate, history = rebuild(task, modified, config.seed,
                                        config.misleading_evidence, config.with_model_advice, config.evidence_file,
                                        frozen_evidence)
    provider = get_provider(config.provider, seed=sampling_seed, temperature=config.temperature)
    policy = POLICIES[config.policy](task=task, provider=provider, with_edges=config.with_edges,
                                     with_model_advice=config.with_model_advice, budget=config.budget)
    proposal = policy.propose(graph, surrogate, history, np.random.default_rng(sampling_seed), len(observations) + 1)
    return proposal, {
        "agent_state": graph.view(len(observations) + 1, with_edges=config.with_edges),
        "provider": provider.metadata() | {"calls": provider.calls, "tokens": provider.tokens,
                                           "latency_s": provider.latency_s},
        "prompt": getattr(policy, "last_prompt", None),
        "intervened_observation_ids": [
            a.id for a, b in zip(modified, observations) if a.value_shown != b.value_shown],
    }


def _distance(task: TaskSpec, a: str, b: str) -> float:
    x = np.array([[task.params_of(cid)[n] for n in task.names] for cid in (a, b)])
    points = task.encode(x)
    return float(np.linalg.norm(points[0] - points[1]))


def paired_interventions(config: Config, rounds: int = 4, repeats: int = 3,
                         direct_llm_only: bool = False,
                         frozen: list[Observation] | None = None,
                         frozen_evidence: list[Claim] | None = None) -> dict:
    """Frozen-prefix, paired-seed next decisions, plus true-vs-true decoding controls."""
    if config.policy != "llm" and direct_llm_only:
        raise ValueError("Direct LLM-only interventions require the LLM policy.")
    if frozen is None:
        base = run_episode(replace(config, feedback="true"), stop_after=rounds)
        observations = list(base.graph.observations.values())
        frozen_evidence = [c for c in base.graph.claims.values() if c.source == "literature"]
    else:
        observations = frozen
    arm_config = replace(config, with_model_advice=not direct_llm_only)
    task, _ = load_task(config.task)
    results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    controls: list[str | None] = []
    for repeat in range(repeats):
        seed = config.seed * 100 + repeat
        for arm in ARMS:
            try:
                proposal, audit = next_decision(arm_config, observations, arm, seed, config.seed,
                                               frozen_evidence=frozen_evidence)
                row = {"candidate_id": proposal.candidate_id, "llm_prediction": proposal.prediction,
                       "rationale": proposal.rationale, "valid": True, **audit}
            except SelectionError as error:
                row = {"candidate_id": None, "valid": False, "failures": error.failures}
            results[arm].append(row)
        try:
            control, _ = next_decision(arm_config, observations, "true", seed + 10000,
                                      frozen_evidence=frozen_evidence)
            controls.append(control.candidate_id)
        except SelectionError:
            controls.append(None)
    summary: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        pairs = [(a["candidate_id"], b["candidate_id"]) for a, b in zip(results["true"], results[arm])
                 if a["valid"] and b["valid"]]
        summary[arm] = {
            "valid_pairs": len(pairs),
            "changed_fraction": float(np.mean([a != b for a, b in pairs])) if pairs else None,
            "mean_scaled_parameter_distance": float(np.mean([_distance(task, a, b) for a, b in pairs])) if pairs else None,
            "physical_parameter_deltas": [
                {n: task.params_of(b)[n] - task.params_of(a)[n] for n in task.names} for a, b in pairs],
        }
    control_pairs = [(a["candidate_id"], b) for a, b in zip(results["true"], controls) if a["valid"] and b]
    variation = float(np.mean([a != b for a, b in control_pairs])) if control_pairs else None
    return {
        "mode": "direct_llm_only" if direct_llm_only else "total_system",
        "config": asdict(arm_config), "frozen_observations": [o.model_dump() for o in observations],
        "arms": results, "summary": summary,
        "decoding_variation_changed_fraction": variation,
        "sensitivity_above_decoding_variation": {
            arm: stats["changed_fraction"] - variation
            if stats["changed_fraction"] is not None and variation is not None else None
            for arm, stats in summary.items()},
        "limitations": [
            "Changed choices are not necessarily improvements; unchanged choices do not prove insensitivity.",
            "Sampling seeds are only sent when EPISTEMIC_SEED_SUPPORTED=true. Even supported seeds need not "
            "guarantee determinism; model revision, stack and server fingerprint are logged.",
            "The influential default target is the best shown prior observation, not causal attribution inferred from prose.",
            "Historical LLM interpretations are discarded in ALL arms; source bundles and observation-derived "
            "claims are rebuilt. This tests fresh re-conditioned choices, not persistence of old rhetoric.",
        ],
    }


def closed_loop(config: Config, seeds: tuple[int, ...] = (0,), initial: int = 3) -> list[dict]:
    """Matched initial candidates/budgets and round-indexed environment seeds; evaluator-only scoring."""
    rows = []
    for seed in seeds:
        task, _ = load_task(config.task)
        rng = np.random.default_rng(seed)
        shared = [task.ids()[int(i)] for i in rng.integers(len(task.ids()), size=initial)]
        episodes = {
            feedback: run_episode(replace(config, seed=seed, feedback=feedback), initial=shared)
            for feedback in ("true", "missing", "corrupted")
        }
        true_regret = episode_metrics(episodes["true"])["best_true_regret"]
        for feedback, episode in episodes.items():
            metrics = episode_metrics(episode)
            rows.append({"seed": seed, **metrics,
                         "regret_loss_vs_true": metrics["best_true_regret"] - true_regret
                         if metrics["best_true_regret"] is not None and true_regret is not None else None,
                         "recovery_to_prefault_regret_plus_0.25": recovery_round(episode, config.fault_round)
                         if config.fault_round else None,
                         "caveat": "After choices diverge, histories differ; this is closed-loop performance, "
                                   "not a frozen-history causal intervention."})
    return rows


def graph_value(config: Config, rounds: int = 8, seeds: tuple[int, ...] = (0,)) -> list[dict]:
    """Both tasks, perfect/perturbed execution, true and controlled misleading evidence, flat/edges."""
    if config.max_searches:
        raise ValueError("Graph controls require matched fixed evidence; disable live searches and use evidence_file.")
    rows = []
    for task_name in ("enzyme", "drug"):
        for execution in ("perfect", "perturbed"):
            for misleading in (False, True):
                for seed in seeds:
                    for edges in (False, True):
                        episode = run_episode(replace(config, task=task_name, seed=seed, budget=rounds,
                                                      policy="llm", with_edges=edges, execution=execution,
                                                      misleading_evidence=misleading))
                        rows.append({"seed": seed, "with_edges": edges, "misleading": misleading,
                                     "prompt_characters": sum(len(s.get("prompt") or "") for s in episode.trajectory),
                                     **episode_metrics(episode)})
    return rows
