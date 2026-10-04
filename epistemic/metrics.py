"""Small, interpretable metrics, always scored against evaluator truth."""

from __future__ import annotations

import numpy as np

from epistemic.loop import Episode
from epistemic.tasks import load_task


def episode_metrics(episode: Episode) -> dict:
    evaluator = episode.evaluator or load_task(episode.config.task)[1]
    task = episode.task
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
    return {
        "task": episode.config.task,
        "acquisition": episode.config.acquisition,
        "evidence_enabled": episode.config.with_evidence,
        "observation_noise": episode.config.observation_noise,
        "execution": episode.config.execution,
        "experiments": len(episode.hidden),
        "completed_budget": len(episode.hidden) == episode.config.budget,
        "stop_reason": episode.stop_reason,
        "elapsed_s": episode.elapsed_s,
        "best_true_value": best,
        "best_true_regret": min((s["regret"] for s in episode.hidden), default=None),
        "final_selection": selection,
        "final_result": episode.final_result(),
        "best_observed_value": (episode.final_result() or {}).get("value_shown"),
        "final_selection_regret": evaluator.regret(selection) if selection else None,
        "final_intended_mean": evaluator.truth(selection) if selection else None,
        "found_optimum": any(s["realised_id"] == evaluator.optimum_id for s in episode.hidden),
        "predictive_mae_vs_truth": float(np.mean(errors)) if errors else None,
        "interval_coverage_vs_truth": float(np.mean(covered)) if covered else None,
        "rejected_evidence_outputs": sum(
            1 for s in episode.trajectory if s["evidence_annotation"]["valid"] is False),
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
        ) + sum(1 for item in episode.initial_evidence.get("searches", [])
                if item.get("status") in ("success", "failed")),
        "active_literature_priors": len(episode.initial_evidence.get("priors", [])),
    }
