"""Small, interpretable metrics, always scored against evaluator truth."""

from __future__ import annotations

import numpy as np

from epistemic.loop import Episode
from epistemic.tasks import load_task


def episode_metrics(episode: Episode) -> dict:
    task, evaluator = load_task(episode.config.task)
    truths = [step["true_value"] for step in episode.hidden]
    best = (max(truths) if task.direction == "maximize" else min(truths)) if truths else None
    selection = episode.final_selection()
    predictions = [(s["round"], s["pre_experiment_prediction"]) for s in episode.trajectory
                   if s.get("pre_experiment_prediction")]
    errors, covered = [], []
    for round, prediction in predictions:
        shown = episode.hidden[round - 1]["true_value"]
        errors.append(abs(shown - prediction["mean"]))
        covered.append(prediction["interval"][0] <= shown <= prediction["interval"][1])
    invalid = sum(1 for s in episode.trajectory if s["llm"]["valid_output"] is False)
    return {
        "task": episode.config.task,
        "policy": episode.config.policy,
        "feedback": episode.config.feedback,
        "execution": episode.config.execution,
        "experiments": len(episode.hidden),
        "completed_budget": len(episode.hidden) == episode.config.budget,
        "best_true_value": best,
        "best_true_regret": min((s["regret"] for s in episode.hidden), default=None),
        "final_selection": selection,
        "final_selection_regret": evaluator.regret(selection) if selection else None,
        "found_optimum": any(s["realised_id"] == evaluator.optimum_id for s in episode.hidden),
        "predictive_mae_vs_truth": float(np.mean(errors)) if errors else None,
        "interval_coverage_vs_truth": float(np.mean(covered)) if covered else None,
        "invalid_llm_selections": invalid,
        "invalid_llm_attempts": sum(len(s["llm"]["failures"]) for s in episode.trajectory),
        "best_true_objective_curve": [
            (max if task.direction == "maximize" else min)(truths[:i + 1]) for i in range(len(truths))],
        "contradictions": len(episode.graph.contradictions()),
        "claim_revisions": sum(len(c.revisions) for c in episode.graph.claims.values()),
        "model_calls": episode.provider_metadata.get("calls", 0),
        "tokens": episode.provider_metadata.get("tokens", 0),
        "latency_s": episode.provider_metadata.get("latency_s", 0.0),
        "literature_searches": sum(
            1 for step in episode.trajectory
            if (step.get("literature_search") or {}).get("status") in ("success", "failed")
        ),
    }


def recovery_round(episode: Episode, fault_round: int, tolerance: float = 0.25) -> int | None:
    """Rounds until a NEW experiment has regret within 0.25 of the pre-fault best true regret.

    Defined before running: a return of None means no recovery within the budget.
    """
    baseline = min((s["regret"] for s in episode.hidden if s["round"] < fault_round), default=None)
    if baseline is None:
        return None
    for step in episode.hidden:
        if step["round"] > fault_round and step["regret"] <= baseline + tolerance:
            return int(step["round"] - fault_round)
    return None
